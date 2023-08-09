import argparse
import datetime
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.common import (ReOrg, Conv)
from models.commond3D import Model3D
from models.experimental import attempt_load
from models.yolo import (IDetect, Detect, IAuxDetect)

from utils.activations import SiLU
from utils.general import (set_logging, check_img_size, colorstr)
from utils.default import MAX_DET
from utils.re_parameteration import Re_parameterization
from utils.torch_utils import select_device

sys.path.append('./')  # to run '$ python *.py' files in subdirectories


def run(**kwargs):
    weight_model = kwargs['weight']
    weight_model = Path(weight_model) if isinstance(weight_model, str) else weight_model
    save_dir = weight_model.parent / weight_model.stem
    if save_dir.exists():
        for _ in save_dir.iterdir():
            _.unlink()
    else:
        save_dir.mkdir()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export your pytorch model to another format")
    parser.add_argument('--weights', nargs='+', type=str, default=['./best.pt'], help='weights path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for onnx export')
    parser.add_argument('--imgsz', type=int, nargs='+', default=-1,
                        help="special input shape, omitting this parameter will use default argument. "
                             "Example --imgsz 640 320 or --imgsz 640")
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic_batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--include', nargs='+', type=str, default='onnx',
                        help='specify a special format for model output')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx for ORT or TRT)')
    parser.add_argument('--max_hw', '--ort', action='store_true', default=None,
                        help='end2end onnxruntime')
    parser.add_argument('--topk_all', type=int, default=MAX_DET,
                        help=f'topk objects for every frames. Default {MAX_DET}"')
    parser.add_argument('--iou_thres', '-iou', type=float, default=0.45, help=f'iou threshold for NMS. Default {0.45}')
    parser.add_argument('--conf_thres', '-conf', type=float, default=0.2, help=f'conf threshold for NMS. Default {0.2}')
    parser.add_argument('--onnx_opset', type=int, default=12,
                        help='onnx opset version, 11 for DmlExecutionProvider. Default 12')
    parser.add_argument('--device', default='cpu', help='cuda:0 or dml:0. default cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include_nms', action='store_true',
                        help='registering EfficientNMS_TRT plugin to export TensorRT engine')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--fp16', '--half', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--v', action='store_true', help='Verbose log')
    parser.add_argument('--author', type=str, default='thnak', help="author's name")
    parser.add_argument('--data', type=str, default='mydataset.yaml', help='data.yaml path')
    parser.add_argument('--trace', action='store_true', help='use torch.jit.trace')
    parser.add_argument('--keras', action='store_true', help='use torch.jit.trace')

    opt = parser.parse_args()
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    set_logging()
    logging.info(f'\n{opt}\n')

    opt.include = [x.lower() for x in opt.include] if isinstance(
        opt.include, list) else [opt.include.lower()]

    torchScript = any(x in ['torchscript', 'coreml', "torchscriptlite"] for x in opt.include)
    torchScriptLite = any(x in ["torchscriptlite"] for x in opt.include)
    ONNX = any(x in ['onnx', 'open', 'openvino'] for x in opt.include)
    openVINO = any(x in ['openvino', 'open'] for x in opt.include)
    tensorFlowjs = any(x in ['tfjs'] for x in opt.include)
    tensorFlowLite = any(x in ['tflite'] for x in opt.include)
    coreML = any(x in ['coreml'] for x in opt.include)
    saved_Model = any(x in ['saved_model', 'tfjs', 'tflite']
                      for x in opt.include)
    graphDef = any(x in ['saved_model', 'grapdef', 'tfjs']
                   for x in opt.include)
    RKNN = any(x in ['rknn'] for x in opt.include)

    t = time.time()
    opt.weights = opt.weights if isinstance(opt.weights, (tuple, list)) else [opt.weights]
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    print(opt.__dict__)

    exPrefix = colorstr('Export:')
    for weight in opt.weights:
        weight = Path(weight)
        logging.info(f'{exPrefix} loading PyTorch model')
        device, gitstatus = select_device(opt.device)
        map_device = 'cpu' if device.type == 'privateuseone' else device
        with torch.no_grad():
            model = attempt_load(weight.as_posix(),
                                 map_location=map_device).to(map_device).eval()  # load FP32 model
            ckpt = torch.load(weight.as_posix(),
                              map_location=map_device)

            for m in model.parameters():
                m.requires_grad = False
            ckpt.pop('model', None)
            ckpt.pop('ema', None)
            ckpt.pop('optimizer', None)
            ckpt.pop('updates', None)
            # prune(model)
        is_3D = isinstance(model, Model3D)

        best_fitness = model.best_fitness if hasattr(model, 'best_fitness') else 0.
        total_image = model.total_image if hasattr(model, 'total_image') else [0]
        gs = int(max(model.stride.max(), 32))  # grid size (max stride)

        input_shape = opt.imgsz
        if not is_3D:
            if input_shape != -1:
                if isinstance(input_shape, (tuple, list)):
                    input_shape = [3, check_img_size(input_shape[0], s=gs),
                                   check_img_size(input_shape[1 if len(input_shape) > 1 else 0], s=gs)]
                else:
                    input_shape = check_img_size(input_shape)
                    input_shape = [3, input_shape, input_shape]
                logging.info(f"{exPrefix} using user input shape {input_shape}")
            else:
                if hasattr(model, "input_shape"):
                    input_shape = model.input_shape
                    logging.info(f"{exPrefix} using input shape from pre-trained model")
                else:
                    input_shape = [3, 640, 640] if model.is_p5() else [3, 1280, 1280]
                    logging.info(
                        f'{exPrefix} using default input shape. to export with special input '
                        f'shape please use --imgsz arg arg')
                if any([tensorFlowjs, tensorFlowLite, saved_Model, graphDef]):
                    input_shape = [3, max(input_shape), max(input_shape)]
                    logging.info(
                        f"{exPrefix} switching to square shape... input_shape: {input_shape}. "
                        f"since some format does not support rectangle shape")
        else:
            input_shape = model.input_shape
            tensorFlowjs = tensorFlowLite = coreML = RKNN = graphDef = saved_Model = openVINO = False
            logging.info(f"{exPrefix} Exporting for Video Classify model. ")
            if opt.imgsz != -1:
                input_shape[2:] = opt.imgsz if len(opt.imgsz) == 2 else [opt.imgsz] * 2

        model_version = model.model_version if hasattr(model, 'model_version') else 0
        model.best_fitness = best_fitness
        model.model_version = model_version
        model.total_image = total_image
        model.input_shape = input_shape
        if not hasattr(model, "is_Classify"):
            model.is_Classify = False
        labels = model.names

        img = torch.zeros(opt.batch_size, *input_shape, device=map_device)

        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, Conv):
                if isinstance(m.act, SiLU):
                    m.act = nn.SiLU()

            if isinstance(m, (Detect, IDetect, IAuxDetect)):
                m.dynamic = opt.dynamic
                if isinstance(m, (IDetect, IAuxDetect)):
                    logging.info(f"{exPrefix} detected training class in the model, trying to re-parameter...")
                    re_paramDir = weight.as_posix().replace(".pt", "_re_param.pt")
                    model = torch.load(weight.as_posix(), map_location=map_device)["model"]
                    model.to(device).eval()
                    model.input_shape = input_shape
                    torch.save({"model": model}, re_paramDir)
                    if Re_parameterization(re_paramDir, re_paramDir):
                        logging.info(f"{exPrefix} re-parameter finished, exporting...\n")
                        ckpt = torch.load(re_paramDir, map_location=map_device)
                        model = ckpt["model"].eval().float().fuse()
                        # end_points_2_break = []
                        for m_ in model.parameters():
                            m_.requires_grad = False
                        # for x, y in model.named_modules():
                        #     end_points_2_break.append(x)
                    else:
                        for m_ in model.parameters():
                            m_.requires_grad = False
                        model = model.fuse()
                        break

# run will put here

        model_Gflops = model.info(verbose=False, img_size=input_shape)
        logging.info(model_Gflops)

        if device.type in ['cuda'] and opt.fp16:
            img = img.to(device).half()
            model = model.to(device).half()
        else:
            if opt.fp16:
                logging.warning(f'Export with fp16 only support for CUDA device, yours {device.type}')

        # model output shape
        y = model(img)
        shape = tuple((y[0] if isinstance(y, (tuple, list)) else y).shape)
        logging.info(f'{exPrefix} model output shape {shape} in pytorch format')
        del y
        # set Detect() layer grid export
        model.model[-1].export = True if any([coreML, is_3D, model.is_Classify]) and not opt.end2end else False
        model.model[-1].include_nms = True if opt.include_nms else False
        model.model[-1].rknn = RKNN

        # metadata
        anchors = anchor_grid = None
        if RKNN:
            if model.is_Classify:
                anchor_grid = model.model[-1].anchor_grid.detach().cpu().numpy().tolist()
                anchors = model.model[-1].anchors.detach().cpu().numpy().tolist()

        MetaData = {'model_infor': model_Gflops,
                    'export_gitstatus': gitstatus,
                    'best_fitness': best_fitness,
                    'nc': len(labels),
                    'stride': model.stride.cpu().tolist(),
                    'names': labels,
                    'total_image': total_image,
                    'export_date': datetime.datetime.now().isoformat('#'),
                    'exporting_opt': vars(opt),
                    "anchor_grid": anchor_grid,
                    "anchors": anchors,
                    "mean": model.yaml.get('mean', [0, 0, 0]),
                    "std": model.yaml.get('std', [1, 1, 1]),
                    "sampling_rate": model.yaml.get("sampling_rate", 0)}
        for index, key in enumerate(ckpt):
            if key == 'model':
                continue
            if key == 'best_fitness':
                ckpt[key] = ckpt[key].tolist()[0] if isinstance(ckpt[key], (np.ndarray, torch.Tensor)) else ckpt[key]
            MetaData[key] = ckpt[key]

        # export
        filenames = []
        if RKNN:
            prefix = colorstr('RKNN:')
            ONNX = True
        # TorchScript export
        if torchScript:
            prefix = colorstr('TorchScript:')
            try:
                from tools.auxexport import TryExportTorchScript

                f = TryExportTorchScript(weight=weight, model=model, feed=img,
                                         logging=logging, MetaData=MetaData,
                                         lite=torchScriptLite,
                                         prefix=prefix)
                logging.info(f'{prefix} export success✅, saved as {f}')
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')
            # CoreML export
        if coreML:
            prefix = colorstr('CoreML:')
            try:
                from tools.auxexport import TryExportCoreML

                f = TryExportCoreML(weight=weight, model=model, feed=img,
                                    map_device=map_device, logging=logging,
                                    prefix=prefix, **opt.__dict__)
                logging.info(f'{prefix} export success✅, saved as %s' % f)
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌: {e}')

        if ONNX:
            prefix = colorstr('ONNX:')
            try:
                from tools.auxexport import TryExport_ONNX

                f = TryExport_ONNX(weight=weight, model=model, feed=img,
                                   map_device=map_device, logging=logging,
                                   rknn=RKNN,
                                   MetaData=MetaData,
                                   prefix=prefix, **opt.__dict__)
                filenames.append(f)
                logging.info(f'{prefix} export success✅, saved as %s' % f)

            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if openVINO:
            prefix = colorstr('OpenVINO:')
            try:
                meta = MetaData
                meta["stride"] = max(MetaData["stride"])
                from tools.auxexport import export_openvino

                logging.info(f'{prefix} Starting export...')
                output_path, _ = export_openvino(file_=weight, metadata=meta, half=True, prefix=prefix)
                logging.info(f'{prefix} export success✅, saved as: {output_path}')
                filenames.append(output_path)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if saved_Model:
            prefix = colorstr('TensorFlow SavedModel:')
            from tools.auxexport import export_saved_model

            f, s_models = export_saved_model(model,
                                             img,
                                             weight,
                                             False,
                                             tf_nms=tensorFlowjs or opt.nms or opt.agnostic_nms,
                                             agnostic_nms=tensorFlowjs or opt.agnostic_nms,
                                             topk_per_class=opt.topk_all,
                                             topk_all=opt.topk_all,
                                             iou_thres=opt.iou_thres,
                                             conf_thres=opt.conf_thres,
                                             keras=opt.keras, prefix=prefix)
            logging.info(f'{prefix} export success✅, saved as {f}')
            filenames.append(f)
            if graphDef:
                prefix = colorstr('TensorFlow GraphDef:')
                try:
                    from tools.auxexport import export_pb

                    f = export_pb(s_models, weight, prefix=prefix)[0]
                    logging.info(f'{prefix} export success✅, saved as {f}')
                    filenames.append(f)
                except Exception as e:
                    logging.info(f'{prefix} export failure❌:\n{e}')

            if tensorFlowLite:
                prefix = colorstr('Tensorflow lite:')
                try:
                    from tools.auxexport import export_tflite, add_tflite_metadata

                    output_path = export_tflite(s_models, img, weight,
                                                int8=opt.int8,
                                                data=opt.data, nms=opt.nms,
                                                agnostic_nms=opt.agnostic_nms,
                                                stride=gs,
                                                prefix=prefix)[0]
                    logging.info(f'{prefix} export success✅, saved as {output_path}')
                    filenames.append(output_path)
                    logging.info(f'{prefix} adding metadata...')
                    meta = MetaData
                    meta["stride"] = max(MetaData["stride"])
                    add_tflite_metadata(output_path, metadata=meta, num_outputs=len(s_models.outputs))
                except Exception as e:
                    logging.info(f'{prefix} export failure❌:\n{e}')

        if tensorFlowjs:
            prefix = colorstr('TensorFlow.js:')
            try:
                from tools.auxexport import export_tfjs

                f = export_tfjs(file_=weight,
                                names=labels,
                                int8=opt.int8,
                                prefix=prefix)
                logging.info(f'{prefix} export success✅, saved as {f}')
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if len(filenames):
            print('\n')
            prefix = colorstr('Export:')
            for i in filenames:
                logging.info(f'{prefix} {i} is exported.')
            logging.info(
                f'\n{prefix} complete (%.2fs). Visualize with https://netron.app/.' % (time.time() - t))
