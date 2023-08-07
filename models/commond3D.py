import argparse
import logging
from copy import deepcopy
from pathlib import Path
import onnx
import thop
import torch
from torch import nn
from utils.general import autopad, make_divisible, check_file, set_logging, fix_problem_with_reuse_activation_funtion
from utils.default import UP_SAMPLE_MODES
from utils.torch_utils import (initialize_weights, model_info, select_device, time_synchronized,
                               fuse_conv_and_bn, fuse_linear_and_bn)
from models.common import Concat, FullyConnected, Conv1D
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Conv3D(nn.Module):
    """Standard convolution"""

    def __init__(self, c1: int, c2: int,
                 k: int | tuple[int, int, int] = 1,
                 s: int | tuple[int, int, int] = 1,
                 p: None | int | tuple[int, int, int] = None,
                 g: int | tuple[int, int, int] = 1,
                 d: int | tuple[int, int, int] = 1,
                 act: any = True,
                 dropout: float = 0.0):
        super(Conv3D, self).__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        k = [k] * 3 if isinstance(k, int) else k
        p = [p] * 3 if isinstance(p, int) else p
        d = [d] * 3 if isinstance(d, int) else d

        pad = [autopad(k_, p_, d_) for k_, p_, d_ in zip(k, p, d)]
        self.conv = nn.Conv3d(c1, c2,
                              kernel_size=k,
                              stride=s,
                              padding=tuple(pad),
                              groups=g if isinstance(g, int) else g[0],
                              dilation=d,
                              bias=False)
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
        act = fix_problem_with_reuse_activation_funtion(act)
        k = (k, k, k) if isinstance(k, int) else k
        p = (p, p, p) if isinstance(p, int) else p
        d = (d, d, d) if isinstance(d, int) else d
        pad = (autopad(k_, p_, d_) for k_, p_, d_ in zip((1, k[1], k[2]), p, d))
        pad2 = (autopad(k_, p_, d_) for k_, p_, d_ in zip((k[0], 1, 1), p, d))
        g1 = g if isinstance(g, int) else g[0]
        self.conv0 = nn.Conv3d(c1, c2,
                               kernel_size=(1, k[1], k[2]),
                               stride=1,
                               padding=pad,
                               groups=g1,
                               dilation=d,
                               bias=False)
        self.bn0 = nn.BatchNorm3d(c2)

        self.conv1 = nn.Conv3d(c2, c2,
                               kernel_size=(k[0], 1, 1),
                               stride=s,
                               padding=pad2,
                               groups=g if isinstance(g, int) else g[0],
                               dilation=d,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(c2)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        inputs = self.conv0(inputs)
        inputs = self.bn0(inputs)
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        return self.drop(self.act(inputs))

    def fuseforward(self, inputs: torch.Tensor):
        return self.act(self.conv1(self.conv0(inputs)))


class MP3D(nn.Module):
    def __init__(self, k=2):
        super(MP3D, self).__init__()
        self.m = nn.MaxPool3d(kernel_size=(1, k, k), stride=(1, k, k))

    def forward(self, x):
        return self.m(x)


class SP3D(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP3D, self).__init__()
        k = (k, k, k) if isinstance(k, int) else k
        s = (s, s, s) if isinstance(s, int) else s
        pad = [x // 2 for x in k]
        self.m = nn.MaxPool3d(kernel_size=k,
                              stride=s,
                              padding=tuple(pad),
                              dilation=1)

    def forward(self, x):
        return self.m(x)


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


class ReOrg3D(nn.Module):
    """https://arxiv.org/pdf/2101.00745.pdf"""

    def __init__(self):
        super(ReOrg3D, self).__init__()

    @staticmethod
    def forward(out):  # x(b, n,c,w,h) -> y(b, n,4c,w/2,h/2)
        out = torch.cat([out[:, :, :, ::2, ::2], out[:, :, :, 1::2, ::2],
                         out[:, :, :, ::2, 1::2], out[:, :, :, 1::2, 1::2]], dim=1)
        return out


# start for  X3D network
class globalsubWay(nn.Module):
    def __init__(self, in_channels: int, out_channel: int,
                 kernel_size: int | tuple[int, int, int] = 1,
                 stride: int | tuple[int, int, int] = 1,
                 pad: int | tuple[int, int, int] = 1, groups: int = 1,
                 dia: int | tuple[int, int, int] = 1):
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
        self.m = nn.Sequential(Conv3D(in_channels, mid_channels, kernel_size, stride, pad,
                                      groups, dia, act=nn.ReLU(), dropout=dropout),
                               Conv3D(mid_channels, mid_channels, 3, 1, 1,
                                      mid_channels, 1, act=nn.SiLU(), dropout=dropout),
                               Conv3D(mid_channels, out_channel, kernel_size, stride, pad,
                                      groups, dia, act=None, dropout=dropout))
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


class residual_block_1(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, act: any):
        super(residual_block_1, self).__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        self.m = nn.Sequential(Conv3D(in_channel, hidden_channel, kernel, 1, [None] * 3, act=act),
                               Conv3D(hidden_channel, hidden_channel, (1, 3, 3), 1, [None] * 3, act=act),
                               Conv3D(hidden_channel, out_channel, 1, 1, [None] * 3, act=False))
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        return self.act(inputs + self.m(inputs))


class residual_block_2(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, hidden_channel: int, kernel: any, stride: any, act: any):
        super(residual_block_2, self).__init__()
        act = fix_problem_with_reuse_activation_funtion(act)
        self.m = nn.Sequential(Conv3D(in_channel, hidden_channel, kernel, 1, [None] * 3, act=act),
                               Conv3D(hidden_channel, hidden_channel, (1, 3, 3), stride, [None] * 3, act=act),
                               Conv3D(hidden_channel, out_channel, 1, 1, [None] * 3, act=False))
        self.m1 = Conv3D(in_channel, out_channel, 1, stride, [None] * 3, act=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, inputs: torch.Tensor):
        return self.act(self.m1(inputs) + self.m(inputs))


class SlowFast(nn.Module):
    export = False

    def __init__(self, in_channel, time_keeper=4, nc=1000, act=nn.ReLU()):
        super(SlowFast, self).__init__()
        self.time_keeper = time_keeper
        slow_output_channel = 64
        fast_to_slow_output_channel = 16
        fast_output_channel = 8
        self.slow_branch_conv0 = nn.Sequential(
            Conv3D(in_channel, slow_output_channel, (1, 7, 7), (1, 2, 2), [None] * 3, act=act),
            nn.MaxPool3d((1, 7, 7), (1, 2, 2), (0, 3, 3)))
        self.fast_branch_conv0 = nn.Sequential(
            Conv3D(in_channel, fast_output_channel, (5, 7, 7), (1, 2, 2), [None] * 3, act=act),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))

        self.conv_fast_to_slow0 = Conv3D(8, fast_to_slow_output_channel, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        slow_output_channel = slow_output_channel + fast_to_slow_output_channel
        self.slow_branch_res0 = residual_block_2(slow_output_channel, 256, 64, [1] * 3, [1] * 3, act=act)
        self.fast_branch_res0 = residual_block_2(fast_output_channel, 32, fast_output_channel, (3, 1, 1), [1] * 3,
                                                 act=act)
        slow_branch_res1 = [residual_block_1(256, 256, 64, [1] * 3, act=act)] * 2
        self.slow_branch_res1 = nn.Sequential(*slow_branch_res1)
        fast_branch_res1 = [residual_block_1(32, 32, 8, (3, 1, 1), act=act)] * 2
        self.fast_branch_res1 = nn.Sequential(*fast_branch_res1)
        # ---
        self.conv_fast_to_slow1 = Conv3D(32, 64, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        self.slow_branch_res2 = residual_block_2(320, 512, 128, [1] * 3, (1, 2, 2), act=act)
        self.fast_branch_res2 = residual_block_2(32, 64, 16, (3, 1, 1), (1, 2, 2), act=act)
        slow_branch_res3 = [residual_block_1(512, 512, 128, [1] * 3, act=act)] * 3
        self.slow_branch_res3 = nn.Sequential(*slow_branch_res3)
        slow_branch_res4 = [residual_block_1(64, 64, 16, (3, 1, 1), act=act)] * 3
        self.fast_branch_res3 = nn.Sequential(*slow_branch_res4)

        # ---
        self.conv_fast_to_slow2 = Conv3D(64, 128, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        self.slow_branch_res4 = residual_block_2(640, 1024, 256, (3, 1, 1), (1, 2, 2), act=act)
        self.fast_branch_res4 = residual_block_2(64, 128, 32, (3, 1, 1), (1, 2, 2), act=act)
        slow_branch_res5 = [residual_block_1(1024, 1024, 256, (3, 1, 1), act=act)] * 5
        self.slow_branch_res5 = nn.Sequential(*slow_branch_res5)
        fast_branch_res5 = [residual_block_1(128, 128, 32, (3, 1, 1), act=act)] * 5
        self.fast_branch_res5 = nn.Sequential(*fast_branch_res5)
        # ---
        self.conv_fast_to_slow3 = Conv3D(128, 256, (7, 1, 1), (4, 1, 1), [None] * 3, act=act)
        # ---
        self.slow_branch_res6 = residual_block_2(1280, 2048, 512, [1] * 3, (1, 2, 2), act=act)
        self.fast_branch_res6 = residual_block_2(128, 256, 64, (3, 1, 1), (1, 2, 2), act=act)
        slow_branch_res7 = [residual_block_1(2048, 2048, 512, (3, 1, 1), act=act)] * 2
        self.slow_branch_res7 = nn.Sequential(*slow_branch_res7)
        fast_branch_res7 = [residual_block_1(256, 256, 64, (3, 1, 1), act=act)] * 2
        self.fast_branch_res7 = nn.Sequential(*fast_branch_res7)
        self.conv = Conv1D(2304, 2480, act=act)
        self.fully = FullyConnected(2480, nc, act=False)
        self.pool = nn.AdaptiveAvgPool3d(4)
        self.post_act = nn.Softmax(1)

    def forward(self, inputs):
        slow_branch, fast_branch = inputs[:, :, ::self.time_keeper, ...], inputs
        slow_branch, fast_branch = self.slow_branch_conv0(slow_branch), self.fast_branch_conv0(fast_branch)
        slow_branch = torch.cat([slow_branch, self.conv_fast_to_slow0(fast_branch)], dim=1)
        # ---
        slow_branch = self.slow_branch_res0(slow_branch)
        fast_branch = self.fast_branch_res0(fast_branch)
        slow_branch = self.slow_branch_res1(slow_branch)
        fast_branch = self.fast_branch_res1(fast_branch)
        # ---
        slow_branch = torch.cat([self.conv_fast_to_slow1(fast_branch), slow_branch], dim=1)
        # # --
        slow_branch = self.slow_branch_res2(slow_branch)
        fast_branch = self.fast_branch_res2(fast_branch)
        slow_branch = self.slow_branch_res3(slow_branch)
        fast_branch = self.fast_branch_res3(fast_branch)
        # ---
        slow_branch = torch.cat([self.conv_fast_to_slow2(fast_branch), slow_branch], dim=1)
        # ---
        slow_branch = self.slow_branch_res4(slow_branch)
        fast_branch = self.fast_branch_res4(fast_branch)
        slow_branch = self.slow_branch_res5(slow_branch)
        fast_branch = self.fast_branch_res5(fast_branch)
        # ---
        slow_branch = torch.cat([self.conv_fast_to_slow3(fast_branch), slow_branch], dim=1)
        # ---
        slow_branch = self.slow_branch_res6(slow_branch)
        fast_branch = self.fast_branch_res6(fast_branch)
        slow_branch = self.slow_branch_res7(slow_branch)
        fast_branch = self.fast_branch_res7(fast_branch)

        slow_branch = self.pool(slow_branch)
        fast_branch = self.pool(fast_branch)
        outputs = torch.cat([slow_branch, fast_branch], dim=1)
        b, c, d, h, w = outputs.shape
        outputs = outputs.view(-1, c, d * h * w)
        outputs = self.conv(outputs)
        outputs = outputs.permute((0, 2, 1))
        outputs = outputs.mean(1)
        outputs = self.fully(outputs)

        if torch.onnx.is_in_onnx_export() or self.export:
            outputs = self.post_act(outputs)

        return outputs


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

    def fuse(self):
        """fuse model Conv2d() + BatchNorm2d() layers, fuse Conv2d + im"""
        prefix = "Fusing layers... "
        pbar = tqdm(self.model.modules(), desc=f'{prefix}', unit=" layer")
        for m in pbar:
            pbar.set_description_str(f"fusing {m.__class__.__name__}")
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
    # model_jit = torch.jit.trace(model, img)
    # torch.jit.save(model_jit, 'jitmodel.pt')
    # model.info(img_size=img.shape[1:], verbose=True)

    y = model(img, profile=True)
    print(y.shape)

    from onnxsim import simplify

    torch.onnx.export(model, torch.rand(1, 3, 32, 256, 256).to(device), "3dmodel.onnx")
    model = onnx.load("3dmodel.onnx")
    model, c = simplify(model)
    onnx.save(model, "3dmodel.onnx")
