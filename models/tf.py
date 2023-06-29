import argparse
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras
from keras.layers import Layer
import logging
import inspect
from typing import Optional
from models.common import *
from models.experimental import MixConv2d, attempt_load
from models.yolo import *
from utils.activations import SiLU, Hardswish
from utils.general import make_divisible, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
logger = logging.getLogger(__name__)


# ROOT = ROOT.relative_to(Path.cwd())  # relative


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    logger.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


class TFShortcut(Layer):
    def __init__(self, dimension=0, w=None):
        super(TFShortcut, self).__init__()
        self.d = dimension

    def __call__(self, x):
        return x[0] + x[1]


class TFRepConv(Layer):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False, w=None):
        super(TFRepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1
        self.act = activations(w.act) if act else tf.identity
        if deploy:
            self.rbr_reparam = TFConv2d(c1, c2, k, s, autopad(k, p), bias=True, w=w.rbr_reparam)
        else:
            self.rbr_identity = (TFBN(w.rbr_indentity) if c2 == c1 and s == 1 else None)
            self.rbr_dense = keras.Sequential(
                [TFConv2d(c1, c2, k, s, g=g, bias=False, w=w.rbr_dense[0]),
                 TFBN(w=w.rbr_dense[1])])
            self.rbr_1x1 = keras.Sequential(
                [TFConv2d(c1, c2, 1, s, g=g, bias=False, w=w.rbr_1x1[0]),
                 TFBN(w.rbr_1x1[1])])

    def __call__(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        a = self.rbr_dense(inputs)
        b = self.rbr_1x1(inputs)
        c = a + b + id_out
        out = self.act(c)
        return out


class TFSPPCSPC(Layer):
    """CSP https://github.com/WongKinYiu/CrossStagePartialNetworks"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), w=None):
        super(TFSPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1, g=g)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2, g=g)
        self.cv3 = TFConv(c_, c_, 3, 1, w=w.cv3, g=g)
        self.cv4 = TFConv(c_, c_, 1, 1, w=w.cv4, g=g)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for i, x in enumerate(k)]
        self.cv5 = TFConv(4 * c_, c_, 1, 1, w=w.cv5, g=g)
        self.cv6 = TFConv(c_, c_, 3, 1, w=w.cv6, g=g)
        self.cv7 = TFConv(2 * c_, c2, 1, 1, w=w.cv7, g=g)

    def __call__(self, inputs):
        x1 = self.cv4(self.cv3(self.cv1(inputs)))
        y1 = self.cv6(self.cv5(tf.concat([x1] + [m(x1) for m in self.m], 3)))
        y2 = self.cv2(inputs)
        return self.cv7(tf.concat((y1, y2), 3))


class TFSPPFCSPC(nn.Module):
    """CSP https://github.com/WongKinYiu/CrossStagePartialNetworks"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(TFSPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, g=g)
        self.cv2 = TFConv(c1, c_, 1, 1, g=g)
        self.cv3 = TFConv(c_, c_, 3, 1, g=g)
        self.cv4 = TFConv(c_, c_, 1, 1, g=g)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="same")
        self.cv5 = TFConv(4 * c_, c_, 1, 1, g=g)
        self.cv6 = TFConv(c_, c_, 3, 1, g=g)
        self.cv7 = TFConv(2 * c_, c2, 1, 1, g=g)

    def __call__(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


def reorg_slice(inputs):
    return tf.concat([inputs[:, ::2, ::2, :],
                      inputs[:, 1::2, ::2, :],
                      inputs[:, ::2, 1::2, :],
                      inputs[:, 1::2, 1::2, :]], 3)


class TFReOrg(Layer):
    def __init__(self, w=None):
        super(TFReOrg, self).__init__()

    def __call__(self, out):  # inputs(b,c,w,h) -> y(b,4c,w/2,h/2)
        out = reorg_slice(out)
        return out


class TFBN(Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        super(TFBN, self).__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.cpu().detach().numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.cpu().detach().numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.cpu().detach().numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.cpu().detach().numpy()),
            epsilon=w.eps)

    def __call__(self, inputs):
        return self.bn(inputs)


class TFMP(Layer):
    def __init__(self, k=2, w=None):
        super(TFMP, self).__init__()
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=k, padding='valid')

    def __call__(self, inputs):
        return self.m(inputs)


class TFSP(Layer):
    def __init__(self, k=3, s=1, w=None):
        super(TFSP, self).__init__()
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=s, padding='SAME')

    def __call__(self, x):
        return self.m(x)


class TFPad(Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def __call__(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class TFConv(Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super(TFConv, self).__init__()
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        conv = keras.layers.Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding='SAME' if s == 1 else 'VALID',
            use_bias=not hasattr(w, 'bn'),
            kernel_initializer=keras.initializers.Constant(
                w.conv.weight.permute(2, 3, 1, 0).cpu().detach().numpy()),
            bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(
                w.conv.bias.cpu().detach().numpy()))

        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def __call__(self, inputs):
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.act(out)
        # return self.act(self.bn(self.conv(inputs)))
        return out


class TFDownC(Layer):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, c1, c2, n=1, k=2, w=None):
        super(TFDownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2 // 2, 3, k, w=w.cv2)
        self.cv3 = TFConv(c1, c2 // 2, 1, 1, w=w.cv3)
        # self.mp = nn.MaxPool2d(kernel_size=k, stride=k)
        self.mp = keras.layers.MaxPool2D(k, k, padding='VALID')

    def __call__(self, x):
        return tf.concat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), 3)


class TFDWConv(Layer):
    """Depthwise convolution"""

    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super(TFDWConv, self).__init__()
        assert c2 % c1 == 0, f'TFDWConv() output={c2} must be a multiple of input={c1} channels'
        conv = keras.layers.DepthwiseConv2D(
            kernel_size=k,
            depth_multiplier=c2 // c1,
            strides=s,
            padding='SAME' if s == 1 else 'VALID',
            use_bias=not hasattr(w, 'bn'),
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).detach().numpy()),
            bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.detach().numpy()))
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity
        self.act = activations(w.act) if act else tf.identity

    def __call__(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class TFDWConvTranspose2d(Layer):
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__()
        assert c1 == c2, f'TFDWConv() output={c2} must be equal to input={c1} channels'
        assert k == 4 and p1 == 1, 'TFDWConv() only valid for k=4 and p1=1'
        weight, bias = w.weight.permute(2, 3, 1, 0).detach().numpy(), w.bias.detach().numpy()
        self.c1 = c1
        self.conv = [
            keras.layers.Conv2DTranspose(filters=1,
                                         kernel_size=k,
                                         strides=s,
                                         padding='VALID',
                                         output_padding=p2,
                                         use_bias=True,
                                         kernel_initializer=keras.initializers.Constant(weight[..., i:i + 1]),
                                         bias_initializer=keras.initializers.Constant(bias[i])) for i in range(c1)]

    def __call__(self, inputs):
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]


class TFFocus(Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(TFFocus, self).__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def __call__(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        # inputs = inputs / 255  # normalize 0-255 to 0-1
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]
        return self.conv(tf.concat(inputs, 3))


class TFBottleneck(Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(TFBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def __call__(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFCrossConv(Layer):
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None):
        super(TFCrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1)
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def __call__(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFConv2d(Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k=1, s=1, g=1, bias=True, w=None):
        super(TFConv2d, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(filters=c2,
                                        kernel_size=k,
                                        strides=s,
                                        padding='SAME' if s == 1 else 'VALID',
                                        use_bias=bias,
                                        kernel_initializer=keras.initializers.Constant(
                                            w.weight.permute(2, 3, 1, 0).detach().detach().cpu().numpy()),
                                        bias_initializer=keras.initializers.Constant(
                                            w.bias.detach().detach().cpu().numpy()) if bias else None)

    def __call__(self, inputs):
        return self.conv(inputs)


class TFBottleneckCSP(Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(TFBottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)
        self.bn = TFBN(w.bn)
        self.act = activations(act=w.act) if hasattr(w, 'act') else tf.identity
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def __call__(self, inputs):
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def __call__(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFC3x(Layer):
    # 3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([
            TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)])

    def __call__(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        super(TFSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in k]

    def __call__(self, inputs):
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        super(TFSPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')

    def __call__(self, inputs):
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFProto(Layer):

    def __init__(self, c1, c_=256, c2=32, w=None):
        super().__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode='nearest')
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, w=w.cv3)

    def __call__(self, inputs):
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):  # warning: all arguments needed including 'w'
        super().__init__()
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2"
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)

    def __call__(self, inputs):
        return self.upsample(inputs)


class TFConcat(Layer):
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None):
        super().__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def __call__(self, inputs):
        return tf.concat(inputs, self.d)


class TFDetect(Layer):
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):  # detection layer
        super().__init__()
        self.stride = tf.convert_to_tensor(w.stride.detach().cpu().numpy(), dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.convert_to_tensor(w.anchors.detach().cpu().numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]), [self.nl, 1, -1, 2])
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def __call__(self, inputs):
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no])

            y = tf.sigmoid(x[i])
            grid = tf.transpose(self.grid[i], [0, 2, 1, 3])
            anchor_grid = tf.repeat(self.anchor_grid[i], repeats=tf.cast(ny * nx, tf.int32), axis=0)

            xy = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]  # xy
            wh = ((y[..., 2:4] * 2) ** 2) * anchor_grid  # wh
            xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
            wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
            y = tf.concat([xy, wh, y[..., 4:5 + self.nc], y[..., 5 + self.nc:]], -1)
            z.append(tf.reshape(y, [-1, self.na * nx * ny, self.no]))
        return (tf.concat(z, 1),)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny), indexing='xy')
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


def parse_model(d, ch, model, imgsz):  # model_dict, input_channels(3)
    logger.info(f"\n{'':>3}{'from':>42}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m_str = m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = a if a in UPSAMPLEMODE else (eval(a) if isinstance(a, str) else a)
            except Exception as ex:
                logger.info(ex)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                 SPP, SPPF, SPPCSPC, SPPFCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m in [Concat, Chuncat]:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m in [Detect, IDetect, IAuxDetect]:
            assert m is Detect, 'IDetect and IAuxDetect is not support, please preparameter'
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            args.append(imgsz)
        else:
            c2 = ch[f]
        tf_m = eval('TF' + m_str.replace('IDetect', 'Detect').replace('nn.', '').replace('IAuxDetect', 'Detect'))
        m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) if n > 1 \
            else tf_m(*args, w=model.model[i])  # module
        torch_m = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        nparam = sum([x.numel() for x in torch_m.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, nparam  # attach index, 'from' index, type, number params
        logger.info('%3s%42s%3s%10.0f  %-40s%-30s' % (i, f, n, nparam, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save)


class TFModel:
    def __init__(self, cfg='cfg/yolov7.yaml', ch=3, nc=None, model=None, imgsz=(640, 640)):  # model, channels, classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

    def __call__(self,
                 inputs,
                 tf_nms=False,
                 agnostic_nms=False,
                 topk_per_class=100,
                 topk_all=100,
                 iou_thres=0.45,
                 conf_thres=0.25):
        y = []  # outputs
        x = inputs
        for index, (m) in enumerate(self.model.layers):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.savelist else None)  # save output

        # Add TensorFlow NMS
        if tf_nms:
            boxes = self._xywh2xyxy(x[0][..., :4])
            probs = x[0][:, :, 4:5]
            classes = x[0][:, :, 5:]
            scores = probs * classes
            if agnostic_nms:
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
            else:
                boxes = tf.expand_dims(boxes, 2)
                nms = tf.image.combined_non_max_suppression(boxes,
                                                            scores,
                                                            topk_per_class,
                                                            topk_all,
                                                            iou_thres,
                                                            conf_thres,
                                                            clip_boxes=False)
            out = (nms,)
            return out
        return x  # output [1,6300,85] = [xywh, conf, class0, class1, ...]

    @staticmethod
    def _xywh2xyxy(xywh):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(Layer):
    # TF Agnostic NMS
    def __init__(self):
        super(AgnosticNMS, self).__init__()

    def call(self, inputs, topk_all, iou_thres, conf_thres):
        # wrap map_fn to avoid TypeSpec related error https://stackoverflow.com/a/65809989/3036450
        return tf.map_fn(lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
                         inputs,
                         fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
                         name='agnostic_nms')

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  # agnostic NMS
        boxes, classes, scores = x
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
        scores_inp = tf.reduce_max(scores, -1)
        selected_inds = tf.image.non_max_suppression(boxes,
                                                     scores_inp,
                                                     max_output_size=topk_all,
                                                     iou_threshold=iou_thres,
                                                     score_threshold=conf_thres)
        selected_boxes = tf.gather(boxes, selected_inds)
        padded_boxes = tf.pad(selected_boxes,
                              paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
                              mode="CONSTANT",
                              constant_values=0.0)
        selected_scores = tf.gather(scores_inp, selected_inds)
        padded_scores = tf.pad(selected_scores,
                               paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                               mode="CONSTANT",
                               constant_values=-1.0)
        selected_classes = tf.gather(class_inds, selected_inds)
        padded_classes = tf.pad(selected_classes,
                                paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                                mode="CONSTANT",
                                constant_values=-1.0)
        valid_detections = tf.shape(selected_inds)[0]
        return padded_boxes, padded_scores, padded_classes, valid_detections


def activations(act=nn.SiLU):
    """Returns TF activation from input PyTorch activation"""
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, (nn.Hardswish, Hardswish)):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, nn.ReLU):
        return lambda x: x * tf.nn.relu(x)
    elif isinstance(act, nn.ReLU6):
        return lambda x: tf.nn.relu6(x)
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)
    elif isinstance(act, nn.Tanh):
        return lambda x: keras.activations.tanh(x)
    elif isinstance(act, nn.Sigmoid):
        return lambda x: keras.activations.sigmoid(x)
    elif isinstance(act, nn.ELU):
        return lambda x: keras.activations.elu(x)
    elif isinstance(act, nn.Hardsigmoid):
        return lambda x: keras.activations.hard_sigmoid(x)
    elif isinstance(act, nn.GELU):
        return lambda x: keras.activations.gelu(x)
    elif isinstance(act, nn.PReLU):
        return lambda x: keras.activations.relu(x, alpha=0.25)
    elif isinstance(act, nn.Softmax):
        return lambda x: keras.activations.softmax(x)
    elif isinstance(act, nn.Softsign):
        return lambda x: keras.activations.softsign(x)
    elif isinstance(act, nn.Softplus):
        return lambda x: keras.activations.softplus(x)
    elif isinstance(act, nn.Softsign):
        return lambda x: keras.activations.softsign(x)
    else:
        raise Exception(f'no matching TensorFlow activation found for PyTorch activation {act}')


def representative_dataset_gen(dataset, ncalib=100):
    """Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays"""
    for n, (path, img, im0s, vid_cap, string, x, y) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255
        yield [im]
        if n >= ncalib:
            break


def run(
        weights=ROOT / 'yolov5s.pt',  # weights path
        imgsz=(640, 640),  # inference size h,w
        batch_size=1,  # batch size
        dynamic=False,  # dynamic batch size
):
    # PyTorch model
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    model = attempt_load(weights, map_location=torch.device('cpu'))
    _ = model(im)  # inference
    model.info()

    # TensorFlow model
    im = tf.zeros((batch_size, *imgsz, 3))  # BHWC image
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    _ = tf_model.predict(im)  # inference

    # Keras model
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()

    logger.info('PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic batch size')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
