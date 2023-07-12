import argparse
import logging
import math
from copy import deepcopy
from pathlib import Path

import onnx
import thop
import torch
from torch import nn, concatenate
from utils.general import autopad, make_divisible, UPSAMPLEMODE, check_file, set_logging
from utils.torch_utils import initialize_weights, scale_img, model_info, select_device
from models.common import Concat
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Conv3D(nn.Module):
    """Standard convolution"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,
                 dropout=0):
        super(Conv3D, self).__init__()
        pad = autopad(k, p)
        self.conv = nn.Conv3d(c1, c2,
                              kernel_size=(1, k, k),
                              stride=(1, s, s),
                              padding=(0, pad, pad),
                              groups=g, dilation=(1, d, d),
                              bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class MP3D(nn.Module):
    def __init__(self, k=2):
        super(MP3D, self).__init__()
        self.m = nn.MaxPool3d(kernel_size=(1, k, k), stride=(1, k, k))

    def forward(self, x):
        return self.m(x)


class SP3D(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP3D, self).__init__()
        self.m = nn.MaxPool3d(kernel_size=(1, k, k),
                              stride=(1, s, s),
                              padding=(0, k // 2, k // 2),
                              dilation=1)

    def forward(self, x):
        return self.m(x)


class Classify3D(nn.Module):
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super(Classify3D, self).__init__()
        self.nc = nc  # number of classes
        list_conv = []
        connected = int(nc * 1.28)

        def calc(x, nc):
            return int(((x + nc) // 2) * 1.28)

        for x in ch:
            a = nn.Sequential(nn.Conv3d(x, calc(x, nc),
                                        kernel_size=(1, 1, 1),
                                        stride=(1, 2, 2),
                                        bias=False),
                              nn.BatchNorm3d(calc(x, nc)),
                              nn.SiLU(),
                              nn.AdaptiveAvgPool3d(1),
                              nn.Flatten(1))
            list_conv.append(a)
            self.m = nn.ModuleList(list_conv)  # output conv
            self.linear = nn.Linear(sum([calc(x, nc) for x in ch]), nc)
            self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i, m in enumerate(self.m):
            out = m(x[i])  # conv
            z.append(out)
        out = concatenate(z, dim=1)
        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        """add softmax activation for deploy"""
        self.linear = nn.Sequential(self.linear, nn.Softmax(1))


class ReOrg3D(nn.Module):
    """https://arxiv.org/pdf/2101.00745.pdf"""

    def __init__(self):
        super(ReOrg3D, self).__init__()

    @staticmethod
    def forward(out):  # x(b, n,c,w,h) -> y(b, n,4c,w/2,h/2)
        out = torch.cat([out[:, :, :, ::2, ::2], out[:, :, :, 1::2, ::2],
                         out[:, :, :, ::2, 1::2], out[:, :, :, 1::2, 1::2]], dim=1)
        return out


def parse_model(d, ch, nc=80):  # model_dict, input_channels(3)
    logger.info('\n%3s%45s%3s%15s  %-50s%-30s' %
                ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d.get('backbone', []) + d.get('head', [])):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = a if a in UPSAMPLEMODE else (eval(a) if isinstance(a, str) else a)
            except Exception as ex:
                logger.error(f'def parse: {ex}')

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv3D]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]

        elif m is nn.BatchNorm3d:
            args = [ch[f]]
        elif m in [Classify3D]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in [Concat]:
            c2 = sum([ch[x] for x in f])
        else:
            c2 = ch[f]

        if m is nn.Upsample:
            if isinstance(args[1], list):
                args[1] = tuple(args[1])

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        nparam = sum([x.numel() for x in m_.parameters()])  # number params
        # index, 'from', type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, nparam
        logger.info('%3s%45s%3s%15s  %-50s%-30s' % (i, f, n_, f"{nparam:,}", t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class Model3D(nn.Module):
    # model, input channels, number of classes
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
            logger.info(
                f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(
                f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], nc=nc)  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.best_fitness = 0.
        self.model_version = 0
        self.total_image = [0]
        self.input_shape = [-1, -1, -1]
        self.reparam = False
        self.is_anchorFree = False
        self.traced = False
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        m.inplace = self.inplace
        self.is_Classify = isinstance(m, Classify3D)
        if self.is_Classify:
            self.stride = torch.tensor([32])
        s = 1024  # scale it up for large shape
        inputSampleShape = [1024] * 2
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi, profile=profile)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs

        for m in self.model:
            if m.f != -1:  # if not from previou layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # initialize biases into Detect(), cf is class frequency
    def _initialize_aux_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # IAuxDetect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.na, -1).detach()  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b2[:, 4] += math.log(8 / (640 / s) ** 2)
            b2[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) %
                  (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):
        """fuse model Conv2d() + BatchNorm2d() layers, fuse Conv2d + im"""
        prefix = "Fusing layers... "
        print(prefix)
        pbar = tqdm(self.model.modules(), desc=f'', unit=" layer")
        for m in pbar:
            if isinstance(m, Classify3D):
                pbar.set_description_str(f"switching to deploy {m.__class__.__name__}")
                m.switch_to_deploy()
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
                        default='yolov7_3D-cls-tiny.yaml', help='model.yaml')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true',
                        help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)[0]

    # Create model
    model = Model3D(opt.cfg, nc=10).to(device)
    model.eval()
    img = torch.rand(1, 3, 4, 224, 224).to(device)
    macs, x = thop.profile(model, inputs=(img,))
    print(f"MACs: {macs/1E9:,} GMACs")
    # y = model(img, profile=True)
    # print(y.shape)

    from onnxsim import simplify

    torch.onnx.export(model, img, "3dmodel.onnx")
    model = onnx.load("3dmodel.onnx")
    model, c = simplify(model)
    onnx.save(model, "3dmodel.onnx")
