from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel
import yaml
import torch
import os
from utils.general import colorstr

def Re_parameterization(inputWeightPath='v7-tiny-training.pt', outputWeightPath = 'cfg/deploy/yolov7.pt', nc = 1, cfgPath='cfg/deploy/yolov7-tiny.yaml'):
    yolov7w_idx= [118,122]
    yolov7e6_idx = [140,144]
    yolov7d6_idx = [162,166]
    yolovye6e_idx = [261,265]
    if os.path.exists(cfgPath):
        print(colorstr('Re-parameteration: ')+f'(Weight: {inputWeightPath}, outputWeight: {outputWeightPath}, numClass: {nc}, cfg: {cfgPath})')
        if 'tiny' in cfgPath:
            total_= '77'
            print(colorstr('Re-parameterizing'),'yolov7-tiny')
        elif cfgPath in 'cfg/deploy/yolov7.yaml':
            total_ = '105'
            print(colorstr('Re-parameterizing'),'yolov7')
        elif cfgPath  in  'cfg/deploy/yolov7x.yaml':
            total_ = '121'
            print(colorstr('Re-parameterizing'),'yolov7x')
        elif cfgPath in 'yolov7-w6.yaml':
            idx = yolov7w_idx[0]
            idx2 = yolov7w_idx[1]
            print(colorstr('Re-parameterizing'),'yolov7-w6')
        elif cfgPath in 'yolov7-e6e.yaml':
            idx = yolovye6e_idx[0]
            idx2 = yolovye6e_idx[1]
            print(colorstr('Re-parameterizing'),'yolov7-e6e')
        elif cfgPath in 'yolov7-e6.yaml':
            idx = yolov7e6_idx[0]
            idx2 = yolov7e6_idx[1]  
            print(colorstr('Re-parameterizing'),'yolov7-e6')          
        elif cfgPath in 'yolov7-d6.yaml':
            idx = yolov7d6_idx[0]
            idx2 = yolov7d6_idx[1]            
            print(colorstr('Re-parameterizing'),'yolov7-d6')   
        
        device = select_device('0'if torch.cuda.is_available() else 'cpu', batch_size=8)[0]
        ckpt = torch.load(inputWeightPath, map_location=device)
        model = Model(cfgPath, ch=3, nc=nc).to(device)
        with open(cfgPath) as f:
            yml = yaml.load(f, Loader=yaml.SafeLoader)
        anchors = len(yml['anchors'][0]) // 2

        # copy intersect weights
        state_dict = ckpt['model'].float().state_dict()
        exclude = []
        intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(intersect_state_dict, strict=False)
        model.names = ckpt['model'].names
        model.nc = ckpt['model'].nc
        
        if 'tiny' in cfgPath  or cfgPath in 'cfg/deploy/yolov7.yaml' or cfgPath  in 'cfg/deploy/yolov7x.yaml':
            for i in range((model.nc+5)*anchors):
                model.state_dict()['model.'+total_+'.m.0.weight'].data[i, :, :, :] *= state_dict['model.'+total_+'.im.0.implicit'].data[:, i, : :].squeeze()
                model.state_dict()['model.'+total_+'.m.1.weight'].data[i, :, :, :] *= state_dict['model.'+total_+'.im.1.implicit'].data[:, i, : :].squeeze()
                model.state_dict()['model.'+total_+'.m.2.weight'].data[i, :, :, :] *= state_dict['model.'+total_+'.im.2.implicit'].data[:, i, : :].squeeze()
            model.state_dict()['model.'+total_+'.m.0.bias'].data += state_dict['model.'+total_+'.m.0.weight'].mul(state_dict['model.'+total_+'.ia.0.implicit']).sum(1).squeeze()
            model.state_dict()['model.'+total_+'.m.1.bias'].data += state_dict['model.'+total_+'.m.1.weight'].mul(state_dict['model.'+total_+'.ia.1.implicit']).sum(1).squeeze()
            model.state_dict()['model.'+total_+'.m.2.bias'].data += state_dict['model.'+total_+'.m.2.weight'].mul(state_dict['model.'+total_+'.ia.2.implicit']).sum(1).squeeze()
            model.state_dict()['model.'+total_+'.m.0.bias'].data *= state_dict['model.'+total_+'.im.0.implicit'].data.squeeze()
            model.state_dict()['model.'+total_+'.m.1.bias'].data *= state_dict['model.'+total_+'.im.1.implicit'].data.squeeze()
            model.state_dict()['model.'+total_+'.m.2.bias'].data *= state_dict['model.'+total_+'.im.2.implicit'].data.squeeze()
        
        else:
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data -= model.state_dict()['model.{}.m.0.weight'.format(idx)].data
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data -= model.state_dict()['model.{}.m.1.weight'.format(idx)].data
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data -= model.state_dict()['model.{}.m.2.weight'.format(idx)].data
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data -= model.state_dict()['model.{}.m.3.weight'.format(idx)].data
            model.state_dict()['model.{}.m.0.weight'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].data
            model.state_dict()['model.{}.m.1.weight'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].data
            model.state_dict()['model.{}.m.2.weight'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].data
            model.state_dict()['model.{}.m.3.weight'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].data
            model.state_dict()['model.{}.m.0.bias'.format(idx)].data -= model.state_dict()['model.{}.m.0.bias'.format(idx)].data
            model.state_dict()['model.{}.m.1.bias'.format(idx)].data -= model.state_dict()['model.{}.m.1.bias'.format(idx)].data
            model.state_dict()['model.{}.m.2.bias'.format(idx)].data -= model.state_dict()['model.{}.m.2.bias'.format(idx)].data
            model.state_dict()['model.{}.m.3.bias'.format(idx)].data -= model.state_dict()['model.{}.m.3.bias'.format(idx)].data
            model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.bias'.format(idx2)].data
            model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.bias'.format(idx2)].data
            model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.bias'.format(idx2)].data
            model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.bias'.format(idx2)].data

            for i in range((model.nc+5)*anchors):
                model.state_dict()['model.{}.m.0.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.0.implicit'.format(idx2)].data[:, i, : :].squeeze()
                model.state_dict()['model.{}.m.1.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.1.implicit'.format(idx2)].data[:, i, : :].squeeze()
                model.state_dict()['model.{}.m.2.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.2.implicit'.format(idx2)].data[:, i, : :].squeeze()
                model.state_dict()['model.{}.m.3.weight'.format(idx)].data[i, :, :, :] *= state_dict['model.{}.im.3.implicit'.format(idx2)].data[:, i, : :].squeeze()
            model.state_dict()['model.{}.m.0.bias'.format(idx)].data += state_dict['model.{}.m.0.weight'.format(idx2)].mul(state_dict['model.{}.ia.0.implicit'.format(idx2)]).sum(1).squeeze()
            model.state_dict()['model.{}.m.1.bias'.format(idx)].data += state_dict['model.{}.m.1.weight'.format(idx2)].mul(state_dict['model.{}.ia.1.implicit'.format(idx2)]).sum(1).squeeze()
            model.state_dict()['model.{}.m.2.bias'.format(idx)].data += state_dict['model.{}.m.2.weight'.format(idx2)].mul(state_dict['model.{}.ia.2.implicit'.format(idx2)]).sum(1).squeeze()
            model.state_dict()['model.{}.m.3.bias'.format(idx)].data += state_dict['model.{}.m.3.weight'.format(idx2)].mul(state_dict['model.{}.ia.3.implicit'.format(idx2)]).sum(1).squeeze()
            model.state_dict()['model.{}.m.0.bias'.format(idx)].data *= state_dict['model.{}.im.0.implicit'.format(idx2)].data.squeeze()
            model.state_dict()['model.{}.m.1.bias'.format(idx)].data *= state_dict['model.{}.im.1.implicit'.format(idx2)].data.squeeze()
            model.state_dict()['model.{}.m.2.bias'.format(idx)].data *= state_dict['model.{}.im.2.implicit'.format(idx2)].data.squeeze()
            model.state_dict()['model.{}.m.3.bias'.format(idx)].data *= state_dict['model.{}.im.3.implicit'.format(idx2)].data.squeeze()            
            
        ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
                'optimizer': None,
                'training_results': None,
                'epoch': -1}
        torch.save(ckpt, outputWeightPath)
        print(colorstr('Re-Parameter:'),f'saved model at:{outputWeightPath} deploy cfg:{cfgPath}')
        
        return True
    else:
        print(f'the arguments is not compatible cfg: {cfgPath}, weight: {inputWeightPath}')
        return False
    