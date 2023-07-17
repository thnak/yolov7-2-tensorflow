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
from utils.torch_utils import initialize_weights, scale_img, model_info, select_device, time_synchronized
from models.common import Concat
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Conv3D(nn.Module):
    """Standard convolution"""

    def __init__(self, c1, c2, k=(1, 1, 1), s=(1, 1, 1), p=(None, None, None), g=(1,), d=(1,), act=True,
                 dropout=0):
        super(Conv3D, self).__init__()
        k = (k, k, k) if isinstance(k, int) else k
        p = (p, p, p) if isinstance(p, int) else p
        pad = (autopad(k_, p_) for k_, p_ in zip(k, p))
        self.conv = nn.Conv3d(c1, c2,
                              kernel_size=k,
                              stride=s,
                              padding=pad,
                              groups=g if isinstance(g, int) else g[0],
                              dilation=d,
                              bias=False)
        # self.conv = Conv2Plus1D(c1, c2, k=k, s=s, p=p, g=g, d=d,act=act, dropout=dropout)
        self.bn = nn.BatchNorm3d(c2)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv3D_2P1(nn.Module):
    """Standard convolution"""

    def __init__(self, c1, c2, k=(1, 1, 1), s=(1, 1, 1), p=(None, None, None), g=(1,), d=(1,), act=True,
                 dropout=0):
        super(Conv3D_2P1, self).__init__()
        pad = (autopad(k_, p_) for k_, p_ in zip(k, p))

        self.conv = Conv2Plus1D(c1, c2, k=k, s=s, p=p, g=g, d=d, act=act, dropout=dropout)
        self.bn = nn.BatchNorm3d(c2)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv2Plus1D(nn.Module):
    def __init__(self, c1, c2, k=(1, 1, 1), s=(1, 1, 1), p=(None, None, None), g=(1,), d=(1,), act=True,
                 dropout=0):
        super(Conv2Plus1D, self).__init__()
        pad = (autopad(k_, p_) for k_, p_ in zip((1, k[1], k[2]), p))
        pad2 = (autopad(k_, p_) for k_, p_ in zip((k[0], 1, 1), p))
        g1 = g if isinstance(g, int) else g[0]
        # g1 = g1 - 1 if c1 % 3 == 0 else g1
        self.conv = nn.Sequential(nn.Conv3d(c1, c2,
                                            kernel_size=(1, k[1], k[2]),
                                            stride=1,
                                            padding=pad,
                                            groups=g1,
                                            dilation=d,
                                            bias=False),
                                  nn.Conv3d(c2, c2,
                                            kernel_size=(k[0], 1, 1),
                                            stride=s,
                                            padding=pad2,
                                            groups=g if isinstance(g, int) else g[0],
                                            dilation=d,
                                            bias=False))

    def forward(self, x):
        return self.conv(x)


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
    def __init__(self, nc=80, dim=2048, ch=(), inplace=True):  # detection layer
        super(Classify3D, self).__init__()
        self.nc = nc  # number of classes
        list_conv = []

        def calc(input_channel, num_classes):
            return int(max(input_channel, num_classes) * 2.048)

        for x in ch:
            a = nn.Sequential(nn.AdaptiveAvgPool3d((1, 2, 4)),
                              nn.Conv3d(x, dim,
                                        kernel_size=(1, 1, 1),
                                        stride=(1, 1, 1),
                                        bias=False),
                              nn.BatchNorm3d(dim),
                              nn.ReLU(),
                              )
            list_conv.append(a)
            self.m = nn.ModuleList(list_conv)  # output conv
            self.linear = nn.Sequential(nn.Linear(sum([dim for _ in ch]), nc, bias=True),
                                        # nn.BatchNorm1d(nc)
                                        )
            self.act = nn.Softmax(dim=1)
            self.m2 = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                    nn.Flatten(1))
            self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i, m in enumerate(self.m):
            out = m(x[i])  # conv
            z.append(out)
        out = concatenate(z, dim=1)
        out = self.m2(out)
        out = self.linear(out)
        if torch.onnx.is_in_onnx_export():
            out = self.act(out)
        return out

    def fuse(self):
        if len(self.linear) > 1:
            linear, bn = self.linear
            output_channel = linear.out_features
            w = linear.weight
            mean = bn.running_mean
            var_sqrt = torch.sqrt(bn.running_var + bn.eps)

            beta = bn.weight
            gamma = bn.bias

            if linear.bias is not None:
                b = linear.bias
            else:
                b = mean.new_zeros(mean.shape)

            w = w * (beta / var_sqrt).reshape([output_channel, 1])
            b = (b - mean) / var_sqrt * beta + gamma
            fused_linear = nn.Linear(linear.in_features,
                                     linear.out_features)

            fused_linear.weight = nn.Parameter(w)
            fused_linear.bias = nn.Parameter(b)
            self.linear = fused_linear


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
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    head_dim = d.get("head_dim", 2048)

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
        if m in [Conv3D, Conv2Plus1D, Conv3D_2P1, subway, ConvPathway1, ConvPathway2]:
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


# start for  X3D network
class globalsubWay(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=1, stride=1, pad=1, groups=1, dia=1):
        super(globalsubWay, self).__init__()
        self.conv = Conv3D(in_channels, out_channel, kernel_size, stride, pad, in_channels, dia, act=None)
        self.m = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                               Conv3D(out_channel, 8, 1, 1, 0, 1, 1, act=nn.ReLU()),
                               Conv3D(8, out_channel, 1, 1, 0, 1, 1, act=nn.Sigmoid()))
        self.post_act = nn.SiLU()

    def forward(self, x):
        out = self.conv(x)
        return self.post_act(out * self.m(out))


class subway(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=1, stride=1, pad=1, groups=1, dia=1, act=True,
                 dropout=0):
        super(subway, self).__init__()
        mid_channels = int(in_channels * 2 + in_channels / 4)
        self.m = nn.Sequential(Conv3D(in_channels, mid_channels, kernel_size, stride, pad, groups, dia, act=nn.ReLU()),
                               Conv3D(mid_channels, mid_channels, 3, 1, 1, mid_channels, 1, act=nn.SiLU()),
                               Conv3D(mid_channels, out_channel, kernel_size, stride, pad, groups, dia, act=None))
        self.post_act = nn.ReLU()

    def forward(self, x):
        return self.post_act(x + self.m(x))


class ConvPathway1(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=1, stride=1, pad=1, groups=1, dia=1, act=True,
                 dropout=0):
        super(ConvPathway1, self).__init__()
        mid_channels = int(in_channels * 2 + in_channels / 4) * kernel_size

        self.conv1 = Conv3D(in_channels, mid_channels, 1, 1, 0, 1, 1, act=nn.ReLU())
        self.res = globalsubWay(mid_channels, mid_channels, 3, (1, 2, 2), 1, 1, 1)
        self.conv3 = Conv3D(mid_channels, out_channel, 1, (1, 1, 1), 0, 1, 1, act=None)

        self.conv2 = Conv3D(in_channels, out_channel, 1, (1, 2, 2), 0, 1, 1, act=None)
        self.post_act = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out1 = self.res(out1)
        out1 = self.conv3(out1)
        return self.post_act(out1 + out2)


class ConvPathway2(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=1, stride=1, pad=1, groups=1, dia=1, act=True,
                 dropout=0):
        super(ConvPathway2, self).__init__()
        mid_channels = int(in_channels * 2 + in_channels / 4)
        self.conv1 = Conv3D(in_channels, mid_channels, 1, 1, 0, 1, 1, act=nn.ReLU())
        self.res = globalsubWay(mid_channels, mid_channels, (3, 3, 3), 1, 1, 1, 1)
        self.conv3 = Conv3D(mid_channels, out_channel, 1, (1, 1, 1), 0, 1, 1, act=None)
        self.post_act = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.res(out1)
        out1 = self.conv3(out1)
        return self.post_act(out1 + x)


# stop for X3D network

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
            if profile:
                c = isinstance(m, Classify3D)
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[
                        0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

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
                m.fuse()
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
                        default='X3D_M.yaml', help='model.yaml')
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
    img = torch.rand(1, 3, 16, 256, 256).to(device)
    macs, x = thop.profile(model, inputs=(img,))
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    print(f"MACs: {macs / 1E9 * 2:,} GMACs, {n_p:,} paramaters")

    y = model(img, profile=True)
    print(y.shape)

    from onnxsim import simplify

    torch.onnx.export(model, img, "3dmodel.onnx")
    model = onnx.load("3dmodel.onnx")
    model, c = simplify(model)
    onnx.save(model, "3dmodel.onnx")
