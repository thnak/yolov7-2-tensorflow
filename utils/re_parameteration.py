from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import is_parallel
import os
from utils.general import colorstr

def Re_parameterization(inputWeightPath='v7-tiny-training.pt',
                        outputWeightPath = 'cfg/deploy/yolov7.pt',
                        device = None):
    prefix = colorstr('Re-parameteration: ')
    model_named = {77: 'YOLOv7-tiny', 105: 'YOLOv7', 121: 'YOLOv7-w6', 122: 'YOLOv7x', 144: 'YOLOv7-e6', 166: 'YOLOv7-d6', 265: 'YOLOv7-e6e'}
    if os.path.exists(inputWeightPath):
        
        ckpt = torch.load(inputWeightPath, map_location=device)
        nodes = len(ckpt['model'].yaml['head']) + len(ckpt['model'].yaml['backbone']) - 1
        p5_model = True if nodes in [77, 105, 121] else False
        if p5_model == False:
            print(f'{prefix} P6 model: {model_named[nodes]}')
            idx = {166: [162,166], 144: [140,144], 265: [261,265], 122: [118,122]}
        else:
            print(f'{prefix} P5 model: {model_named[nodes]}')
        nc = ckpt['model'].nc
        model_recfg = eval(str(ckpt['model'].yaml).replace('IDetect', 'Detect').replace('IAuxDetect', 'Detect'))
        model = Model(model_recfg, ch=3, nc=nc).to(device).float()
        anchors = len(ckpt['model'].model[-1].anchor_grid.squeeze()[0])
        # d6:: 166, e6:: 144, e6e:: 265, x:: 122
        state_dict = ckpt['model'].to(device).float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = nc
        
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
            
        ckpt['model'] = deepcopy(model.module if is_parallel(model) else model)
        ckpt['epoch'] = -1
        torch.save(ckpt, outputWeightPath)
        print(f'{prefix}saved model at: {outputWeightPath}')
        
        return True
    else:
        print(f'{prefix}File not found weight: {inputWeightPath}')
        return False
    