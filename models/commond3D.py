import argparse
import logging
from copy import deepcopy
from pathlib import Path
import onnx
import thop
import torch
from torch import nn

from models.customs_models import SlowFast
from utils.general import make_divisible, check_file, set_logging
from utils.default import UP_SAMPLE_MODES
from utils.torch_utils import (initialize_weights, model_info, select_device, time_synchronized,
                               fuse_conv_and_bn, fuse_linear_and_bn)
from models.common import Concat, FullyConnected, Conv1D, Conv3D, Conv2Plus1D, ReOrg3D, subway, \
    ConvPathway1, ConvPathway2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Classify3D(nn.Module):
    export = False

    def __init__(self, nc=80, dim=2048, ch=(), inplace=True):  # detection layer
        super(Classify3D, self).__init__()
        self.nc = nc  # number of classes
        list_conv = []
        for _ in ch:
            a = nn.Sequential(nn.AdaptiveAvgPool3d(4))
            list_conv.append(a)
        self.m = nn.ModuleList(list_conv)  # output conv
        self.conv = Conv1D(sum([_ for _ in ch]), dim, 1, 1, 0, 1, 1, act=nn.LeakyReLU())
        self.linear = FullyConnected(dim, nc, act=False)
        self.act = nn.Softmax(dim=1)
        self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i, m in enumerate(self.m):
            z.append(m(x[i]))

        out = torch.cat(z, dim=1)
        b, c, d, h, w = out.shape
        out = out.view(-1, c, d * h * w)
        out = self.conv(out)
        out = out.permute((0, 2, 1))
        out = out.mean(1)
        out = self.linear(out)
        if torch.onnx.is_in_onnx_export() or self.export:
            out = self.act(out)
        return out


# start for  X3D network


# stop for X3D network

class Model3D(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):
        super(Model3D, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = self.parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.best_fitness = 0.
        self.model_version = 0
        self.total_image = [0]
        self.input_shape = [-1, -1, -1]
        self.reparam = False
        self.is_anchorFree = False
        self.traced = False
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        m.inplace = self.inplace
        self.is_Classify = True
        self.stride = torch.tensor([8])
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False):
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:

            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                c = isinstance(m, Classify3D)
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0]
                o /= 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                shape = x.shape if not isinstance(x, list) else [_.shape for _ in x]
                print(shape)

                for _ in range(10):
                    m(x.copy() if c else x)
                t1 = time_synchronized() - t
                dt.append(t1 * 100)
                print('%3i%10.6f%10.0f%10.1fms %-40s' % (m.i, o, m.np, dt[-1], m.type))
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    @staticmethod
    def parse_model(d, ch):
        logger.info('\n%3s%45s%3s%15s  %-50s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
        nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
        head_dim = d.get("head_dim", 2048)
        low_frame_rate = d.get("low_frame_rate", 1)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

        # from, number, module, args
        for i, (f, n, m, args) in enumerate(d.get('backbone', []) + d.get('head', [])):
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = a if a in UP_SAMPLE_MODES else (eval(a) if isinstance(a, str) else a)
                except Exception as ex:
                    logger.error(f'def parse: {ex}')

            n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [Conv3D, Conv2Plus1D, subway, ConvPathway1, ConvPathway2]:
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]

                for x in range(2, 7):
                    args[x] = tuple(args[x]) if isinstance(args[x], list) else args[x]
                    args[x] = args[x] * 3 if len(args[x]) == 1 else args[x]
                args[4] = (None, None, None) if "None" in args[4] else args[4]  # pads
                args[5] = args[5][0] if isinstance(args[5], tuple) else args[5]  # groups
                if m is ConvPathway1:
                    args[2] = args[2][0] if isinstance(args[2], tuple) else args[2]
            elif m is nn.BatchNorm3d:
                args = [ch[f]]
            elif m in [Classify3D]:
                args.append([ch[x] for x in f])
                if isinstance(args[2], int):  # number of anchors
                    args[2] = [list(range(args[2] * 2))] * len(f)
            elif m in [Concat]:
                c2 = sum([ch[x] for x in f])
            elif m is nn.Upsample:
                if isinstance(args[1], list):
                    args[1] = tuple(args[1])
            elif m in [SlowFast]:
                c1, c2 = ch[f], args[0]
                args = [c1, c2, *args[1:]]

            else:
                c2 = ch[f]

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            n_param = sum([x.numel() for x in m_.parameters()])  # number params
            # index, 'from', type, number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, n_param
            logger.info('%3s%45s%3s%15s  %-50s%-30s' % (i, f, n_, f"{n_param:,}", t, args))  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)

    def fuse(self):
        """fuse model Conv2d() + BatchNorm2d() layers, fuse Conv2d + im"""
        prefix = "Fusing layers... "
        p_bar = tqdm(self.model.modules(), desc=f'{prefix}', unit=" layer")
        for m in p_bar:
            p_bar.set_description_str(f"fusing {m.__class__.__name__}")
            if isinstance(m, (Conv3D, Conv1D)):
                if hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    m.forward = m.fuseforward
                    delattr(m, 'bn')
                    if hasattr(m, "drop"):
                        delattr(m, "drop")
            elif isinstance(m, FullyConnected):
                if hasattr(m, 'bn'):
                    m.linear = fuse_linear_and_bn(m.linear, m.bn)
                    m.forward = m.fuseforward
                    delattr(m, 'bn')
            elif isinstance(m, Conv2Plus1D):
                if all([hasattr(m, "bn0"), hasattr(m, "bn1")]):
                    m.conv0 = fuse_conv_and_bn(m.conv0, m.bn0)
                    m.conv1 = fuse_conv_and_bn(m.conv1, m.bn1)
                    m.forward = m.fuseforward
                    delattr(m, 'bn0')
                    delattr(m, 'bn1')
                    if hasattr(m, "drop"):
                        delattr(m, "drop")

        return self

    def info(self, verbose=False, img_size=640):  # print model information
        return model_info(self, verbose, img_size)

    def is_p5(self):
        for m in self.model.modules():
            if isinstance(m, ReOrg3D):
                return False
        else:
            return True

    def num_nodes(self):
        return len(self.yaml['backbone']) + len(self.yaml['head']) - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='yolov7-cls-tiny.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true',
                        help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)[0]

    # Create model
    model = Model3D(opt.cfg, nc=101).to(device)
    model.eval()
    model.fuse()
    img = torch.rand(1, 3, 32, 224, 224).to(device)
    model_jit = torch.jit.trace(model, img)
    torch.jit.save(model_jit, 'jitmodel.pt')
    model.info(img_size=img.shape[1:], verbose=True)
    #
    # y = model(img, profile=True)
    # print(y.shape)
    #
    # from onnxsim import simplify
    #
    # torch.onnx.export(model, torch.rand(1, 3, 32, 256, 256).to(device), "3dmodel.onnx")
    # model = onnx.load("3dmodel.onnx")
    # model, c = simplify(model)
    # onnx.save(model, "3dmodel.onnx")
    # print('export finished')
