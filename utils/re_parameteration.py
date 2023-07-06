from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import is_parallel
import os
from utils.general import colorstr
import argparse
from pathlib import Path

def Re_parameterization(inputWeightPath='v7-tiny-training.pt',
                        outputWeightPath='yolov7.pt',
                        device=None):
    prefix = colorstr('Re-parameteration: ')
    model_named = {-1: 'unnamed', 77: 'YOLOv7-tiny', 105: 'YOLOv7', 121: 'YOLOv7-w6', 122: 'YOLOv7x', 144: 'YOLOv7-e6',
                   166: 'YOLOv7-d6', 265: 'YOLOv7-e6e'}
    idx = {166: [162, 166], 144: [140, 144], 265: [261, 265], 122: [118, 122]}
    device = "cpu" if device is None else device
    if os.path.exists(inputWeightPath):
        ckpt = torch.load(inputWeightPath, map_location=device)
        old_model = ckpt['model'].eval()
        nc = old_model.nc
        p5_model = old_model.is_p5()
        nodes = old_model.num_nodes()
        string_cfg = str(old_model.yaml).replace('IDetect', 'Detect').replace('IV6Detect', 'V6Detect')
        if "IAuxDetect" in string_cfg:
            while 1:
                named = model_named[nodes] if nodes in model_named else model_named[-1]
                named = named.lower()
                print()
                print(f"Maybe your model is {named}")
                input_cfg = input(f"Please put your cfg deploy compatible with your model "
                                  f"here.\nExample: type 'cfg/deploy/{named}.yaml' if you "
                                  f"train with 'cfg/training/{named}.yaml'.\n"
                                  f"Say no to ignore re-paramater.\n")
                if input_cfg.lower() == "no":
                    Path(inputWeightPath).unlink(missing_ok=True)
                    return False
                input_cfg = Path(input_cfg)
                if not input_cfg.exists():
                    print(f"File '{input_cfg.as_posix()}' not found.")
                else:
                    try:
                        import yaml
                        with open(input_cfg.as_posix(), "r") as f:
                            string_cfg = yaml.load(f, yaml.SafeLoader)
                            break
                    except Exception as ex:
                        print(f"error\n{ex}")
                print()

        cfg = eval(string_cfg) if isinstance(string_cfg, str) else string_cfg
        if 'head_deploy' in cfg:
            cfg['head'] = cfg['head_deploy']
            cfg.pop('head_deploy', None)
        model = Model(cfg, ch=3, nc=nc).to(device=device, dtype=torch.float32).eval()

        imgsz = old_model.input_shape
        total_image = old_model.total_image if hasattr(old_model, "total_image") else [0]
        model_version = old_model.model_version if hasattr(old_model, "model_version") else 0
        best_fitness = old_model.best_fitness if hasattr(old_model, "best_fitness") else 0.

        model.best_fitness = best_fitness
        model.input_shape = imgsz
        model.total_image = total_image
        model.model_version = model_version
        model.reparam = True
        model.info(verbose=True, img_size=imgsz)
        print(f'{prefix}{"P5" if p5_model else "P6"} branch; named model: {model_named[nodes] if nodes in model_named else model_named[-1]}')
        # anchors = len(old_model.model[-1].anchor_grid.squeeze()[0])
        anchors = len(model.yaml['anchors'][0]) // 2
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
                for index, x in enumerate(model.model[-1].m):
                    model.state_dict()[f'model.{nodes}.m.{index}.weight'].data[i, :, :, :] *= state_dict[f'model.{nodes}.im.{index}.implicit'].data[:, i, ::].squeeze()

            for i, m in enumerate(model.model[-1].m):
                bias_key = f"model.{nodes}.m.{i}.bias"
                model.state_dict()[bias_key].data += state_dict[f'model.{nodes}.m.{i}.weight'].mul(state_dict[f'model.{nodes}.ia.{i}.implicit']).sum(1).squeeze()
                model.state_dict()[bias_key].data *= state_dict[f'model.{nodes}.im.{i}.implicit'].data.squeeze()

        else:
            for index, x in enumerate(model.model[-1].m):
                weight_key_0 = f"model.{idx[nodes][0]}.m.{index}.weight"
                weight_key_1 = f"model.{idx[nodes][1]}.m.{index}.weight"
                bias_key_0 = f'model.{idx[nodes][0]}.m.{index}.bias'
                bias_key_1 = f"model.{idx[nodes][1]}.m.{index}.bias"
                model.state_dict()[weight_key_0].data -= model.state_dict()[weight_key_0].data
                model.state_dict()[weight_key_0].data += state_dict[weight_key_1].data
                model.state_dict()[bias_key_0].data -= model.state_dict()[bias_key_0].data
                model.state_dict()[bias_key_0].data += state_dict[bias_key_1].data

            for i in range((model.nc+5)*anchors):
                for index, x in enumerate(model.model[-1].m):
                    weight_key_0 = f'model.{idx[nodes][0]}.m.{index}.weight'
                    implicit_key = f'model.{idx[nodes][1]}.im.{index}.implicit'
                    model.state_dict()[weight_key_0].data[i, :, :, :] *= state_dict[implicit_key].data[:, i, : :].squeeze()
            for index, x in enumerate(model.model[-1].m):
                weight_key_1 = f"model.{idx[nodes][1]}.m.{index}.weight"
                bias_key_0 = f'model.{idx[nodes][0]}.m.{index}.bias'
                implicit_key = f'model.{idx[nodes][1]}.im.{index}.implicit'
                implicit_key_1 = f"model.{idx[nodes][1]}.ia.{index}.implicit"
                model.state_dict()[bias_key_0].data += state_dict[weight_key_1].mul(state_dict[implicit_key_1]).sum(1).squeeze()
                model.state_dict()[bias_key_0].data *= state_dict[implicit_key].data.squeeze()

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
