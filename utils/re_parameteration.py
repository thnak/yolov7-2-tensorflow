from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import is_parallel
import os
from utils.general import colorstr
import argparse


def Re_parameterization(inputWeightPath='v7-tiny-training.pt',
                        outputWeightPath='cfg/deploy/yolov7.pt',
                        device=None):
    prefix = colorstr('Re-parameteration: ')
    model_named = {-1: 'unnamed', 77: 'YOLOv7-tiny', 105: 'YOLOv7', 121: 'YOLOv7-w6', 122: 'YOLOv7x', 144: 'YOLOv7-e6',
                   166: 'YOLOv7-d6', 265: 'YOLOv7-e6e'}
    idx = {166: [162, 166], 144: [140, 144], 265: [261, 265], 122: [118, 122]}
    if os.path.exists(inputWeightPath):
        ckpt = torch.load(inputWeightPath, map_location=device)
        old_model = ckpt['model'].eval()
        nc = old_model.nc
        p5_model = old_model.is_p5()
        nodes = old_model.num_nodes()
        string_cfg = str(old_model.yaml).replace('IDetect', 'Detect').replace('IV6Detect', 'V6Detect')
        cfg = eval(string_cfg)
        if 'head_deploy' in cfg:
            cfg['head'] = cfg['head_deploy']
            cfg.pop('head_deploy', None)
        model = Model(cfg, ch=3, nc=nc).to(device=device, dtype=torch.float32).eval()

        imgsz = old_model.input_shape
        total_image = old_model.total_image
        model_version = old_model.model_version
        best_fitness = old_model.best_fitness

        model.best_fitness = best_fitness
        model.input_shape = imgsz
        model.total_image = total_image
        model.model_version = model_version
        model.reparam = True
        model.info(verbose=True, img_size=imgsz)
        print(f'{prefix}{"P5" if p5_model else "P6"} branch; named model: {model_named[nodes] if nodes in model_named else model_named[-1]}')
        anchors = len(old_model.model[-1].anchor_grid.squeeze()[0])
        # d6:: 166, e6:: 144, e6e:: 265, x:: 122
        state_dict = old_model.to(device).float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if
                                k in model.state_dict() and not any(x in k for x in exclude) and v.shape ==
                                model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = old_model.names
        model.nc = nc
        del old_model

        if p5_model:
            for i in range((model.nc+5)*anchors):
                model.state_dict()[f'model.{nodes}.m.0.weight'].data[i, :, :, :] *= state_dict[f'model.{nodes}.im.0.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{nodes}.m.1.weight'].data[i, :, :, :] *= state_dict[f'model.{nodes}.im.1.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{nodes}.m.2.weight'].data[i, :, :, :] *= state_dict[f'model.{nodes}.im.2.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{nodes}.m.0.bias'].data += state_dict[f'model.{nodes}.m.0.weight'].mul(state_dict[f'model.{nodes}.ia.0.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{nodes}.m.1.bias'].data += state_dict[f'model.{nodes}.m.1.weight'].mul(state_dict[f'model.{nodes}.ia.1.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{nodes}.m.2.bias'].data += state_dict[f'model.{nodes}.m.2.weight'].mul(state_dict[f'model.{nodes}.ia.2.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{nodes}.m.0.bias'].data *= state_dict[f'model.{nodes}.im.0.implicit'].data.squeeze()
            model.state_dict()[f'model.{nodes}.m.1.bias'].data *= state_dict[f'model.{nodes}.im.1.implicit'].data.squeeze()
            model.state_dict()[f'model.{nodes}.m.2.bias'].data *= state_dict[f'model.{nodes}.im.2.implicit'].data.squeeze()

        else:
            model.state_dict()[f'model.{idx[nodes][0]}.m.0.weight'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.0.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.1.weight'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.1.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.2.weight'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.2.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.3.weight'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.3.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.0.weight'].data += state_dict[f'model.{idx[nodes][1]}.m.0.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.1.weight'].data += state_dict[f'model.{idx[nodes][1]}.m.1.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.2.weight'].data += state_dict[f'model.{idx[nodes][1]}.m.2.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.3.weight'].data += state_dict[f'model.{idx[nodes][1]}.m.3.weight'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.0.bias'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.0.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.1.bias'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.1.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.2.bias'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.2.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.3.bias'].data -= model.state_dict()[f'model.{idx[nodes][0]}.m.3.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.0.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.0.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.1.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.1.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.2.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.2.bias'].data
            model.state_dict()[f'model.{idx[nodes][0]}.m.3.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.3.bias'].data

            for i in range((model.nc+5)*anchors):
                model.state_dict()[f'model.{idx[nodes][0]}.m.0.weight'].data[i, :, :, :] *= state_dict[f'model.{idx[nodes][1]}.im.0.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{idx[nodes][0]}.m.1.weight'].data[i, :, :, :] *= state_dict[f'model.{idx[nodes][1]}.im.1.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{idx[nodes][0]}.m.2.weight'].data[i, :, :, :] *= state_dict[f'model.{idx[nodes][1]}.im.2.implicit'].data[:, i, : :].squeeze()
                model.state_dict()[f'model.{idx[nodes][0]}.m.3.weight'].data[i, :, :, :] *= state_dict[f'model.{idx[nodes][1]}.im.3.implicit'].data[:, i, : :].squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.0.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.0.weight'].mul(state_dict[f'model.{idx[nodes][1]}.ia.0.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.1.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.1.weight'].mul(state_dict[f'model.{idx[nodes][1]}.ia.1.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.2.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.2.weight'].mul(state_dict[f'model.{idx[nodes][1]}.ia.2.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.3.bias'].data += state_dict[f'model.{idx[nodes][1]}.m.3.weight'].mul(state_dict[f'model.{idx[nodes][1]}.ia.3.implicit']).sum(1).squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.0.bias'].data *= state_dict[f'model.{idx[nodes][1]}.im.0.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.1.bias'].data *= state_dict[f'model.{idx[nodes][1]}.im.1.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.2.bias'].data *= state_dict[f'model.{idx[nodes][1]}.im.2.implicit'].data.squeeze()
            model.state_dict()[f'model.{idx[nodes][0]}.m.3.bias'].data *= state_dict[f'model.{idx[nodes][1]}.im.3.implicit'].data.squeeze()

        with torch.no_grad():
            ckpt['model'] = deepcopy(model.module if is_parallel(model) else model).to('cpu')
            for m in ckpt['model'].parameters():
                m.requires_grad = False
            imgsz = ckpt['model'].input_shape
            ckpt['model'].info(verbose=True, img_size=imgsz)
            input_sample = torch.zeros(1, *imgsz, device='cpu', requires_grad=False)
            y = ckpt['model'](input_sample)
            ckpt['model'].half()
            ckpt['epoch'] = -1
            torch.save(ckpt, outputWeightPath)
            print(f'{prefix}saved model at: {outputWeightPath}')
            return True
    else:
        print(f'{prefix}File not found weight: {inputWeightPath}')
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='model for pre-paramter')
    parser.add_argument('--output-path', type=str, help='model path output')
    opt = parser.parse_args()
    result = Re_parameterization(opt.input_path, opt.output_path, device='cpu')
