from utils.plots import plot_one_box
import numpy as np
from utils.general import check_requirements
from utils.loss import SigmoidBin
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.general import make_divisible, check_file, set_logging, colorstr, xywh2xyxy, box_iou
from utils.autoanchor import check_anchor_order
from utils.activations import *
from models.experimental import *
from models.common import *
import torchvision
import torch
import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

UPSAMPLEMODE = ['nearest', 'linear', 'bilinear', 'bicubic']


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False
    dynamic = False  # https://github.com/WongKinYiu/yolov7/pull/1270

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(
            self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv
        self.inplace = inplace

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    # new xy
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(
            [torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return box, score


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False
    dynamic = False

    def __init__(self, nc=80, anchors=(), ch=(), ignore=None):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float()
        self.register_buffer('anchors', a.view(self.nl, -1, 2))  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)
                    # new xy
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))
                    wh = wh ** 2 * (4 * self.anchor_grid[i].detach())  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(
                c1, c2), self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return box, score


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=(), ignore=None):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[:self.nl])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch[self.nl:])  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.nl])
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch[:self.nl])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            x[i + self.nl] = self.m2[i](x[i + self.nl])
            x[i + self.nl] = x[i + self.nl].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].detach())  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1), x[:self.nl])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].detach()  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z,)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return box, score


class V6Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max(ch[0] // 4, 16), max(ch[0], self.no - 4)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        if self.training:
            return x, box, cls
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in self.make_anchors(x, self.stride, 0.5))
            self.shape = shape

        dbox = self.dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, (x, box, cls))

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = torch.split(distance, 2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class IV6Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max(ch[0] // 4, 16), max(ch[0], self.no - 4)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max)

        self.ia2 = nn.ModuleList(ImplicitA(x) for x in ch)
        self.ia3 = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im2 = nn.ModuleList(ImplicitM(4 * self.reg_max) for _ in ch)
        self.im3 = nn.ModuleList(ImplicitM(self.nc) for _ in ch)

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            # x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            x[i] = torch.cat((self.im2[i](self.cv2[i](self.ia2[i](x[i]))), self.im3[i](self.cv3[i](self.ia3[i](x[i])))),
                             1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        if self.training:
            return x, box, cls
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in self.make_anchors(x, self.stride, 0.5))
            self.shape = shape

        dbox = self.dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, (x, box, cls))

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = torch.split(distance, 2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (5 objects and 80 classes per 640 image)


class Model(nn.Module):
    # model, input channels, number of classes
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
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
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.best_fitness = 0.
        self.model_version = 0
        self.total_image = [0]
        self.input_shape = [-1, -1, -1]
        self.reparam = False
        self.inplace = self.yaml.get('inplace', True)
        self.use_anchor = False
        # Build strides, anchors
        m = self.model[-1]  # Detect()
        m.inplace = self.inplace
        s = 256
        if isinstance(m, (Detect, IDetect)):
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            self.use_anchor = True

        elif isinstance(m, IAuxDetect):
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_aux_biases()  # only run once
            self.use_anchor = True

        elif isinstance(m, (V6Detect, IV6Detect)):
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[0]])  # forward
            self.stride = m.stride
            m.bias_init()
            self.use_anchor = False

        # Init weights, biases
        initialize_weights(self)

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
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
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [
                    x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect))
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
        m = self.model[-1]  # Detect() module
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

    def _print_weights(self):
        for m in self.model.modules():
            if type(m) is Bottleneck:
                print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                # print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                # print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names',
                                    'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        return model_info(self, verbose, img_size)

    def is_p5(self, nodes=None):
        if not nodes:
            nodes = len(self.yaml['backbone']) + len(self.yaml['head']) - 1
        elif 'p5' in self.yaml:
            return eval(self.yaml['p5']) if isinstance(self.yaml['p5'], str) else self.yaml['p5']
        out = nodes in [77, 105, 121]
        return out

    def num_nodes(self):
        return len(self.yaml['backbone']) + len(self.yaml['head']) - 1


class ONNX_Engine(object):
    """ONNX Engine class for inference with onnxruntime"""

    def __init__(self, ONNX_EnginePath='',
                 confThres=0.25, iouThres=0.45,
                 classes_nms=None, agnostic_nms=False, multi_label_nms=False,
                 max_det_nms=300, maxWorkSpace=2, prefix='', ):

        self.prefix = prefix or colorstr('ONNX engine:')
        self.confThres = confThres
        self.iouThres = iouThres
        self.classes_nms = classes_nms
        self.agnostic_nms = agnostic_nms
        self.multi_label_nms = multi_label_nms
        self.max_det_nms = max_det_nms

        import platform
        import torch

        self.device = torch.device('cpu')
        nvidia_GPUDevices = torch.cuda.is_available()
        is_x64 = True if '64' in platform.architecture()[0] else False
        assert is_x64, f'{prefix} not support for 32bit device'
        operating_system = platform.system()
        amd_GPUdevices = True if 'AMD' in platform.machine() else False
        nvidia_GPUDevices = torch.cuda.is_available()
        intel_Devicess = True if 'Intel' in platform.machine() else False
        cpu_device = False
        if operating_system in ['Windows']:
            check_requirements(['onnxruntime-directml'])
        elif operating_system == 'Linux':
            if is_x64 and nvidia_GPUDevices:
                check_requirements(['onnxruntime-gpu'])
            elif is_x64 and intel_Devicess:
                check_requirements(['onnxruntime-openvino'])
            elif is_x64:
                cpu_device = True
                check_requirements(['onnxruntime'])

        import onnxruntime as onnxrt
        self.runTime = onnxrt
        self.GB = maxWorkSpace * 1024 * 1024 * 1024

        try:
            import onnx
            onnx_model = onnx.load(ONNX_EnginePath)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            del onnx_model
        except onnx.checker.ValidationError as e:
            logger.error(f"{prefix} The model is invalid: {e}")
            exit()

        # self.providers = [
        #     ('TensorrtExecutionProvider',
        #      {
        #         'device_id': 0,
        #         'trt_max_workspace_size': self.GB,
        #         'trt_fp16_enable': True,
        #         'trt_dla_enable': True
        #         }),
        #     ('CUDAExecutionProvider',
        #      {
        #         'device_id': 0,
        #         'arena_extend_strategy': 'kNextPowerOfTwo',
        #         'cudnn_conv1d_pad_to_nc1d': True,
        # 'cudnn_conv_use_max_workspace': '2',
        #         'gpu_mem_limit': self.GB,
        #         'cudnn_conv_algo_search': 'EXHAUSTIVE',
        #         'do_copy_in_default_stream': True,
        #         'enable_cuda_graph': True}),
        #     'CPUExecutionProvider'] if torch.cuda.is_available() else self.runTime.get_available_providers()
        self.providers = self.runTime.get_available_providers()

        session_opt = self.runTime.SessionOptions()
        session_opt.enable_profiling = False
        session_opt.log_severity_level = 3
        session_opt.use_deterministic_compute = True
        session_opt.enable_mem_pattern = False if 'DmlExecutionProvider' in self.providers else True
        session_opt.graph_optimization_level = self.runTime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opt.execution_mode = self.runTime.ExecutionMode.ORT_PARALLEL if cpu_device else self.runTime.ExecutionMode.ORT_SEQUENTIAL

        self.session = self.runTime.InferenceSession(ONNX_EnginePath, sess_options=session_opt,
                                                     providers=self.providers)
        self.session.enable_fallback()

        self.imgsz = self.session.get_inputs()[0].shape[2:]
        self.imgsz = self.imgsz if isinstance(self.imgsz[0], int) else [640, 640]
        self.batch_size = self.session.get_inputs()[0].shape[0]
        self.batch_size = 0 if self.batch_size == 'batch' else self.batch_size
        self.dynamic_batch = self.batch_size == 0
        self.half = self.session.get_inputs()[0].type == "tensor(float16)"

        self.output_names = [x.name for x in self.session.get_outputs()]
        self.input_names = [i.name for i in self.session.get_inputs()]

        self.names, self.nc, self.stride, self.rectangle = None, None, None, False
        meta = self.session.get_modelmeta().custom_metadata_map

        self.best_fitness = meta['best_fitness'] if 'best_fitness' in meta else -1.
        self.best_fitness = self.best_fitness if isinstance(self.best_fitness, (float, int)) else -1
        self.export_gitstatus = str(meta['export_gitstatus']) if 'export_gitstatus' in meta else 'unknow'
        self.stride = int(meta['stride']) if 'stride' in meta else 32
        self.names = eval(meta['names']) if 'names' in meta else ['nonamed'] * 1000
        self.nc = int(meta['nc']) if 'nc' in meta else len(self.names)
        self.export_date = str(meta['export_date']) if 'export_date' in meta else 'YYYY-MM-DD#hh:mm:ss.0000'
        self.exporting_opt = eval(meta['exporting_opt']) if 'exporting_opt' in meta else None
        self.model_version = eval(meta['model_version']) if 'model_version' in meta else -1
        self.training_results = str(meta['training_results']) if 'training_results' in meta else '-1'
        self.wandb_id = str(meta['wandb_id']) if 'wandb_id' in meta else '---'
        self.train_gitstatus = str(meta['train_gitstatus']) if 'train_gitstatus' in meta else '-'
        self.export_gitstatus = str(meta['export_gitstatus']) if 'export_gitstatus' in meta else '-'
        self.rectangle = self.imgsz[0] != self.imgsz[1]
        self.isLandscape = self.imgsz[0] < self.imgsz[1]  # h,c
        self.is_end2end = self.exporting_opt['max_hw'] and self.exporting_opt['end2end']
        self.p5 = 'p5_model' in meta

    def get_infor(self):
        print(f'{self.prefix}\n{vars(self)}')

    def preproc_for_infer(self, im):
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        im = im.cpu().numpy()
        return im

    @staticmethod
    def concat(im):
        if len(im) > 1:
            im = np.concatenate(im)
        else:
            im = im[0]
        return im

    def infer(self, im):
        y = self.session.run(self.output_names, {self.input_names[0]: im})
        if self.is_end2end:
            return y
        else:
            if isinstance(y, (list, tuple)):
                self.prediction = self.from_numpy(y[0]) if len(
                    y) == 1 else [self.from_numpy(x) for x in y]
                return self.non_max_suppression()
            else:
                self.prediction = self.from_numpy(y)
                return self.non_max_suppression()

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    @staticmethod
    def xyxy2xywh(x, dim):
        y = [0] * 4
        y[0] = ((x[0] + x[2]) / 2) / dim[1]  # x center
        y[1] = ((x[1] + x[3]) / 2) / dim[0]  # y center
        y[2] = (x[2] - x[0]) / dim[1]  # width
        y[3] = (x[3] - x[1]) / dim[0]  # height
        return y

    def end2end(self, outputs, ori_images, dwdh, ratio, fps, bfc, frames=0):
        image = [None] * len(ori_images)
        bbox = [None] * len(ori_images)
        txtcolor2, bboxcolor2 = bfc.getval(index=0)
        if isinstance(dwdh, list):
            dwdhs = dwdh
            ratios = ratio
        else:
            dwdhs = [dwdh]
            ratios = [ratio]

        for index, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            if image[int(batch_id)] is None:
                image[int(batch_id)] = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdhs[int(batch_id)] * 2)
            box /= ratios[int(batch_id)][0]
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 2)
            if score < self.confThres:
                continue
            name = self.names[cls_id]
            name += ' ' + str(score)
            bbox[int(batch_id)] = self.xyxy2xywh(box, image[int(batch_id)].shape[:2])
            txtcolor, bboxcolor = bfc.getval(index=cls_id)
            image[int(batch_id)] = plot_one_box(box, image[int(batch_id)],
                                                txtColor=txtcolor,
                                                bboxColor=bboxcolor, label=name, frameinfo=[])

        for index in range(len(image)):
            if image[index] is None:
                image[index] = plot_one_box(None, ori_images[index],
                                            txtColor=txtcolor2,
                                            bboxColor=bboxcolor2, label=None,
                                            frameinfo=[f'FPS: {fps}',
                                                       f'Total object: {int(len(outputs) / len(ori_images))}', frames])
            else:
                image[index] = plot_one_box(None, image[index],
                                            txtColor=txtcolor2,
                                            bboxColor=bboxcolor2, label=None,
                                            frameinfo=[f'FPS: {fps}',
                                                       f'Total object: {int(len(outputs) / len(ori_images))}', frames])
        return image, bbox

    def warmup(self, num=10):
        imgsz = (self.batch_size, self.session.get_inputs()[0].shape[1], self.imgsz[0], self.imgsz[1])
        im = np.ones(imgsz, dtype=np.float16 if self.half else np.float32)
        if self.session.get_providers()[0] in ['CUDAExecutionProvider', 'TensorrtExecutionProvider',
                                               'DmlExecutionProvider']:
            t0 = time_synchronized()
            logger.info(f'\n{self.prefix} warming up... image shape {im.shape}\n')
            for _ in range(num):
                self.infer(im)
            return ((time_synchronized() - t0) / num) / self.batch_size

    def non_max_suppression(self):
        # prediction = self.prediction
        # conf_thres = self.confThres
        # classes = self.classes_nms
        # agnostic = self.agnostic_nms
        # multi_label = self.multi_label_nms
        # max_det = self.max_det_nms

        # Checks
        assert 0 <= self.confThres <= 1, f'{self.prefix} Invalid Confidence threshold {self.confThres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iouThres <= 1, f'{self.prefix} Invalid IoU {self.iouThres}, valid values are between 0.0 and 1.0'
        if isinstance(self.prediction, (list, tuple)):
            # select only inference output
            self.prediction = self.prediction[0]

        device = self.prediction.device
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            self.prediction = self.prediction.cpu()
        bs = self.prediction.shape[0]  # batch size
        nc = self.prediction.shape[2] - 5  # number of self.classes_nms
        xc = self.prediction[..., 4] > self.confThres  # candidates

        # Settings
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        # multiple labels per box (adds 0.5ms/img)
        self.multi_label_nms &= nc > 1
        merge = False  # use merge-NMS

        t = time_synchronized()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6), device=self.prediction.device)] * bs
        for xi, x in enumerate(self.prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            # center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if self.multi_label_nms:
                i, j = (x[:, 5:mi] > self.confThres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None],
                               j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[
                    conf.view(-1) > self.confThres]

            # Filter by class
            if self.classes_nms is not None:
                x = x[(x[:, 5:6] == torch.tensor(
                    self.classes_nms, device=x.device)).any(1)]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            # sort by confidence and remove excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            # Batched NMS
            c = x[:, 5:6] * (0 if self.agnostic_nms else max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, self.iouThres)  # NMS
            i = i[:self.max_det_nms]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > self.iouThres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time_synchronized() - t) > time_limit:
                logger.warning(
                    f'{self.prefix} WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded
        return output


class TensorRT_Engine(object):
    """TensorRT using for TensorRT inference
    only available on Nvidia's devices
    """

    def __init__(self, TensorRT_EnginePath, confThres=0.5, iouThres=0.45, prefix=''):

        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        from utils.general import BackgroundForegroundColors
        import yaml
        import cv2
        import os
        self.cuda = cuda
        self.trt = trt

        self.yaml = yaml
        self.cv2 = cv2
        self.os = os
        self.prefix = colorstr(f'TensorRT engine:')
        self.confThres = confThres
        self.iouThes = iouThres
        logger_ = self.trt.Logger(self.trt.Logger.WARNING)
        logger_.min_severity = self.trt.Logger.Severity.ERROR
        runtime = self.trt.Runtime(logger_)
        self.trt.init_libnvinfer_plugins(
            logger_, '')  # initialize TensorRT plugins

        try:
            if os.path.exists('mydataset.yaml'):
                with open('mydataset.yaml', 'r') as dataset_cls_name:
                    data_ = self.yaml.load(
                        dataset_cls_name, Loader=self.yaml.SafeLoader)
                    self.nc = data_['nc']
                    self.names = data_['names']
            else:
                self.nc = 999
                self.names = [i for i in range(self.nc)]
            with open(TensorRT_EnginePath, "rb") as f:
                serialized_engine = f.read()
        except IOError:
            logging.warning(f'Error: {IOError}, the item is required')
            exit()
        self.Colorselector = BackgroundForegroundColors(self.names)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = self.cuda.Stream()

        for binding in engine:
            size = self.trt.volume(engine.get_binding_shape(binding))
            dtype = self.trt.nptype(engine.get_binding_dtype(binding))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        """inference an image

        Args:
            img (image): _description_

        Returns:
            num, final_boxes, final_scores, final_cls_inds
        """
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            self.cuda.memcpy_htod_async(
                inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(
                out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]
        return data

    def detect_video(self, video_path, video_outputPath='', end2end=False, noSave=True):
        """detect objection from video"""
        video_outputPath = self.os.path.join(video_outputPath, 'results.mp4')
        from utils.ffmpeg_ import FFMPEG_recorder
        if not self.os.path.exists(video_path):
            logging.info(f'{self.prefix} video not found, exiting')
            exit()
        cap = self.cv2.VideoCapture(video_path)
        fps = int(round(cap.get(self.cv2.CAP_PROP_FPS)))
        width = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))

        if not noSave:
            logging.info(f'{self.prefix} Save video at: {video_outputPath}')
            ffmpeg = FFMPEG_recorder(video_outputPath, (width, height), fps)
        fps, avg = 0, []
        timeStart = time_synchronized()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = self.preproc(frame, self.imgsz)
            t1 = time_synchronized()
            data = self.infer(blob)
            fps = (fps + (1. / (time_synchronized() - t1))) / 2
            avg.append(fps)
            frame = self.cv2.putText(
                frame, "FPS:%d " % fps, (0, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            t2 = time_synchronized()
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[
                                                             :num[0]].reshape(-1, 1),
                                       np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(
                    data, (1, -1, int(5 + self.n_classes)))[0]
                dets = self.postprocess(predictions, ratio)
            logging.info(f'{self.prefix} FPS: {round(fps, 3)}, ' +
                         f'nms: {round(time_synchronized() - t2, 3)}' if end2end else 'postprocess:' + f' {round(time_synchronized() - t2, 3)}')
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                frame = self.vis(frame, final_boxes, final_scores,
                                 final_cls_inds, names=self.names)
            if not noSave:
                ffmpeg.writeFrame(frame)
        if not noSave:
            ffmpeg.stopRecorder()
        cap.release()

        logging.info(f'{self.prefix} Finished! ' + f'save at {video_outputPath} ' if not noSave else '' +
                                                                                                     f'total {round(time_synchronized() - timeStart, 2)} second, avg FPS: {round(sum(avg) / len(avg), 3)}')

    def inference(self, origin_img, end2end=False):
        """ detect single image
            Return: image
        """
        img_, ratio = self.preproc(origin_img, self.imgsz)
        t1 = time_synchronized()
        data = self.infer(img_)
        logging.info(f'speed: {time_synchronized() - t1}s')
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[
                                                         :num[0]].reshape(-1, 1),
                                   np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5 + self.n_classes)))[0]
            dets = self.postprocess(predictions, ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = self.vis(origin_img, final_boxes,
                                  final_scores, final_cls_inds,
                                  names=self.names)
        return origin_img

    @staticmethod
    def postprocess(self, predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores)
        return dets

    def get_fps(self):
        """Warming up and calculate fps"""
        img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        t1 = self.time.perf_counter()
        avgT = []
        for _ in range(20):
            _ = self.infer(img)
            t1 = self.time.perf_counter() - t1
            avgT.append(t1)
        logging.info(
            f'{self.prefix} Warming up with {(sum(avgT) / len(avgT) / 10)}FPS (etc)')

    def nms(self, boxes, scores):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iouThes)[0]
            order = order[inds + 1]
        return keep

    def multiclass_nms(self, boxes, scores):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > self.confThres
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def preproc(self, image, input_size, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = self.cv2.resize(img, (int(img.shape[1] * r), int(
            img.shape[0] * r)), interpolation=self.cv2.INTER_LINEAR, ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def vis(self, img, boxes, scores, cls_ids, names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < self.confThres:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            text = '{}:{:.2f}'.format(names[cls_id], score)
            txt_color, txt_bk_color = self.Colorselector.getval(cls_id)
            txt_size = self.cv2.getTextSize(
                text, self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            self.cv2.rectangle(img, (x0, y0), (x1, y1), txt_bk_color, 2)
            c1, c2 = (x0, y0), (x1, y1)
            c2 = c1[0] + txt_size[0], c1[1] - txt_size[1] - 3
            self.cv2.drawContours(img, [np.array([(c1[0] + txt_size[0], c1[1] - txt_size[1] - 3), (c1[0] +
                                                                                                   txt_size[0], c1[1]),
                                                  (c1[0] + txt_size[0] + txt_size[1] + 3, c1[1])])], 0, txt_bk_color,
                                  -1, 16)
            self.cv2.rectangle(img, c1, c2, txt_bk_color, - 1, self.cv2.LINE_AA)  # filled
            self.cv2.putText(img, text, (c1[0], c1[1] - 2), 0, 0.4,
                             txt_color, thickness=1, lineType=self.cv2.LINE_AA)
        return img


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' %
                ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = a if a in UPSAMPLEMODE else eval(a) if isinstance(a, str) else a
            except Exception as ex:
                logger.error(f'ex: {ex}')

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
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC, C3, C2f]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, SPPFCSPC, GhostSPPCSPC,
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
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m in [IV6Detect, V6Detect]:
            args.append([ch[x] for x in f])
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        # assert c2 < 1024, f'torch 1.13.1 max channel size is 1024, yours {torch.__version__}'
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        nparam = sum([x.numel() for x in m_.parameters()])  # number params
        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, nparam
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, nparam, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(
            f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='yolor-csp-c.yaml', help='model.yaml')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true',
                        help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)[0]

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)
