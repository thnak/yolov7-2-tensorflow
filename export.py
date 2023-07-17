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
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.common import (ReOrg, Conv)
from models.commond3D import Model3D
from models.experimental import attempt_load, End2End
from models.yolo import (IDetect, Detect, IAuxDetect)

from utils.activations import SiLU
from utils.general import (set_logging, check_img_size, check_requirements, colorstr, ONNX_OPSET, ONNX_OPSET_TARGET,
                           MAX_DET)
from utils.re_parameteration import Re_parameterization
from utils.torch_utils import select_device

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export your pytorch model to another format")
    parser.add_argument('--weights', nargs='+', type=str, default=['./best.pt'], help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for onnx export')
    parser.add_argument('--imgsz', type=int, nargs='+', default=-1,
                        help="special input shape, omitting this parameter will use default argument. "
                             "Example --imgsz 640 320 or --imgsz 640")
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--include', nargs='+', type=str, default='onnx',
                        help='specify a special format for model output')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx for ORT or TRT)')
    parser.add_argument('--max-hw', '--ort', action='store_true', default=None,
                        help='end2end onnxruntime')
    parser.add_argument('--topk-all', type=int, default=MAX_DET,
                        help=f'topk objects for every frames. Default {MAX_DET}"')
    parser.add_argument('--iou-thres', '-iou', type=float, default=0.45, help=f'iou threshold for NMS. Default {0.45}')
    parser.add_argument('--conf-thres', '-conf', type=float, default=0.2, help=f'conf threshold for NMS. Default {0.2}')
    parser.add_argument('--onnx-opset', type=int, default=12,
                        help='onnx opset version, 11 for DmlExecutionProvider. Default 12')
    parser.add_argument('--device', default='cpu', help='cuda:0 or dml:0. default cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true',
                        help='registering EfficientNMS_TRT plugin to export TensorRT engine')
    parser.add_argument('--nms', action='store_true', help='TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help='TF: add agnostic NMS to model')
    parser.add_argument('--fp16', '--half', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--v', action='store_true', help='Verbose log')
    parser.add_argument('--author', type=str, default='Nguyễn Văn Thạnh', help="author's name")
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

    torchScript = any(x in ['torchscript', 'coreml'] for x in opt.include)
    torchScriptLite = any(x in ['torchscriptlite'] for x in opt.include)
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
    opt.weights = [Path(x) for x in opt.weights] if isinstance(
        opt.weights, (tuple, list)) else [Path(opt.weights)]
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)

    exPrefix = colorstr('Export:')

    for weight in opt.weights:
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
                        f'{exPrefix} using default input shape. to export with special input shape please use --imgsz arg arg')
                if any([tensorFlowjs, tensorFlowLite, saved_Model, graphDef]):
                    input_shape = [3, max(input_shape), max(input_shape)]
                    logging.info(
                        f"{exPrefix} switching to square shape... input_shape: {input_shape}. since some format does not support rectangle shape")
        else:
            input_shape = model.input_shape
            tensorFlowjs = tensorFlowLite = coreML = RKNN = graphDef = saved_Model = openVINO = False
            ONNX = True
            logging.info(f"{exPrefix} Exporting for Video Classify model. ")

        model_version = model.model_version if hasattr(model, 'model_version') else 0
        model.best_fitness = best_fitness
        model.model_version = model_version
        model.total_image = total_image
        model.input_shape = input_shape
        labels = model.names

        img = torch.zeros(opt.batch_size, *input_shape, device=map_device)

        # Update model
        end_points_2_break = []
        start_points_2_break = 0
        for k, m in model.named_modules():
            end_points_2_break.append(k)
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, ReOrg):
                start_points_2_break = []
            if isinstance(m, Conv):
                if isinstance(start_points_2_break, list):
                    if not len(start_points_2_break):
                        start_points_2_break.append(f'/model.0/Concat_output_0')
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
                        end_points_2_break = []
                        for m_ in model.parameters():
                            m_.requires_grad = False
                        for x, y in model.named_modules():
                            end_points_2_break.append(x)
                    else:
                        for m_ in model.parameters():
                            m_.requires_grad = False
                        model = model.fuse()
                        break

        if RKNN:
            end_points_2_break = end_points_2_break[-len(model.yaml['anchors']):]
            for i, x in enumerate(end_points_2_break):
                from_modul, modul_idx, attr, idx = x.split('.')
                end_points_2_break[i] = f'/{from_modul}.{modul_idx}/{attr}.{idx}/Conv_output_0'

        model_Gflops = model.info(verbose=False, img_size=input_shape)
        logging.info(model_Gflops)
        model.model[-1].export = True if any([coreML]) and not opt.end2end else False  # set Detect() layer grid export,
        # for coreml export set to True
        y = model(img)

        if device.type in ['cuda'] and opt.fp16:
            img = img.to(device).half()
            model = model.to(device).half()
        else:
            if opt.fp16:
                logging.warning(f'Export with fp16 only support for CUDA device, yours {device.type}')

        # model output shape
        shape = tuple((y[0] if isinstance(y, (tuple, list)) else y).shape)
        logging.info(f'{exPrefix} model output shape {shape} in pytorch format')

        if opt.include_nms:
            model.model[-1].include_nms = True
            y = None
        filenames = []
        if RKNN:
            prefix = colorstr('RKNN:')
            if isinstance(start_points_2_break, int):
                start_points_2_break = ['images']
            logging.info(f'{prefix} input name: {start_points_2_break} output name: {end_points_2_break}')
            ONNX = True
        # TorchScript export
        if torchScript:
            prefix = colorstr('TorchScript:')
            try:
                logging.info(
                    f'\n{prefix} Starting TorchScript export with torch{torch.__version__}')
                f = weight.as_posix().replace('.pt', '.torch-script.pt')  # filename
                ts = torch.jit.trace(model, img, strict=True, check_trace=True)
                ts.save(f)
                logging.info(f'{prefix} export success✅, saved as {f}')
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')
            # CoreML export
        if coreML:
            prefix = colorstr('CoreML:')
            try:
                check_requirements('coremltools')
                from models.yolo import iOSModel
                import coremltools as ct

                logging.info(f'\n{prefix} Starting CoreML export with coremltools {ct.__version__}')
                if opt.end2end:
                    ts = iOSModel(model, img)
                    ts = torch.jit.trace(ts, img, strict=False)
                ct_model = ct.convert(ts,
                                      inputs=[ct.ImageType('image',
                                                           shape=img.shape,
                                                           scale=1 / 255.0,
                                                           bias=[0, 0, 0])])
                bits, mode = (8, 'kmeans_lut') if opt.int8 else (
                    16, 'linear') if opt.fp16 else (32, None)
                if bits < 32:
                    if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=DeprecationWarning)
                            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(
                                ct_model, bits, mode)
                    else:
                        logging.info(
                            f'{prefix} quantization only supported on macOS, skipping...')
                f = weight.as_posix().replace('.pt', '.mlmodel')  # filename
                ct_model.save(f)
                logging.info(
                    f'{prefix} CoreML export success✅, saved as %s' % f)
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} CoreML export failure❌: {e}')
        if torchScriptLite:
            prefix = colorstr('TorchScript-Lite:')
            try:
                logging.info(
                    f'\n{prefix} Starting TorchScript-Lite export with torch {torch.__version__}')
                f = weight.as_posix().replace('.pt', '.torchscript.ptl')  # filename
                tsl = torch.jit.trace(model, img, strict=False)
                tsl = optimize_for_mobile(tsl)
                tsl._save_for_lite_interpreter(f)
                logging.info(
                    f'{prefix} TorchScript-Lite export success✅, saved as {f}')
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')
        if ONNX:
            prefix = colorstr('ONNX:')
            check_requirements(('onnx', 'onnxmltools'))
            import onnx
            import onnxmltools

            logging.info(
                f'\n{prefix} Starting ONNX export with onnx {onnx.__version__}')
            f = weight.as_posix().replace('.pt', '.onnx')  # filename
            output_names = ['classes',
                            'boxes'] if y is None else ['output']
            if y is None:
                output_names = ["classes", "boxes"]
            else:
                output_names = ["output"]
            dynamic_axes = None
            if opt.dynamic:
                dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                'output': {0: 'batch', 2: 'y', 3: 'x'}}
            if opt.dynamic_batch:
                opt.batch_size = 'batch'
                dynamic_axes = {
                    'images': {
                        0: 'batch',
                    }, }
                if opt.end2end and not opt.max_hw:
                    output_axes = {
                        'num_dets': {0: 'batch'},
                        'det_boxes': {0: 'batch'},
                        'det_scores': {0: 'batch'},
                        'det_classes': {0: 'batch'}, }
                else:
                    output_axes = {'output': {0: 'batch'}, }
                dynamic_axes.update(output_axes)
            if opt.end2end:
                x = 'TensorRT' if not opt.max_hw else 'ONNXRUNTIME'
                logging.info(f'{prefix} Starting export end2end model for {colorstr(x)}')
                model = End2End(model, opt.topk_all, opt.iou_thres,
                                opt.conf_thres, max(input_shape[1:]) if opt.max_hw else None, map_device, len(labels))
                if opt.end2end and not opt.max_hw:
                    output_names = ['num_dets', 'det_boxes',
                                    'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True
            if opt.onnx_opset not in ONNX_OPSET:
                logging.info(f'{prefix} onnx opset must be in {ONNX_OPSET}, switching to 12')
                opt.onnx_opset = 12
            dml = False
            try:
                import torch_directml

                dml = True
            except ImportError:
                pass
            except Exception:
                pass
            if opt.onnx_opset not in ONNX_OPSET_TARGET and dml:
                logging.info(
                    f'{prefix} onnx opset tested for version {ONNX_OPSET_TARGET}, newer version may have poor performance for ONNXRUNTIME in DmlExecutionProvider')
            torch.onnx.disable_log()
            if img.dtype != torch.float16 and opt.trace:
                model = torch.jit.trace(model, img).eval()
                logging.info(f'{prefix} Traced model!')
            torch.onnx.export(model,
                              img, f, verbose=opt.v,
                              opset_version=opt.onnx_opset,
                              input_names=['images'],
                              output_names=output_names,
                              training=torch.onnx.TrainingMode.EVAL,
                              dynamic_axes=dynamic_axes,
                              keep_initializers_as_inputs=True)
            if RKNN:
                onnx.utils.extract_model(input_path=f, output_path=f, input_names=start_points_2_break,
                                         output_names=end_points_2_break)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            if opt.end2end and not opt.max_hw:
                for i in onnx_model.graph.output:
                    for j in i.type.tensor_type.shape.dim:
                        j.dim_param = str(shapes.pop(0))

            if opt.simplify:
                try:
                    check_requirements('onnxsim')
                    import onnxsim

                    logging.info(f'{prefix} Starting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    logging.info(f'{prefix} Simplifier failure❌: {e}')

            onnx.checker.check_model(onnx_model)  # check onnx model
            logging.info(f'{prefix} writing metadata for model...')
            if hasattr(model, "is_Classify"):
                anchor_grid = model.model[
                    -1].anchor_grid.detach().cpu().numpy().tolist() if not model.is_Classify else None
                anchors = model.model[-1].anchors.detach().cpu().numpy().tolist() if not model.is_Classify else None
            else:
                anchor_grid = model.model[-1].anchor_grid.detach().cpu().numpy().tolist()
                anchors = model.model[-1].anchors.detach().cpu().numpy().tolist()
            onnx_MetaData = {'model_infor': model_Gflops,
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
                             }
            key_prefix = colorstr('yellow', 'key:')
            for index, key in enumerate(ckpt):
                if key == 'model':
                    continue
                if key == 'best_fitness':
                    ckpt[key] = ckpt[key].tolist()[0] if isinstance(ckpt[key], (np.ndarray, torch.Tensor)) else ckpt[
                        key]
                onnx_MetaData[key] = ckpt[key]

            for index, key in enumerate(onnx_MetaData):
                metadata = onnx_model.metadata_props.add()
                metadata.key = key
                metadata.value = str(onnx_MetaData[key])
                # logging.info(f'{key_prefix} {key}, value: {onnx_MetaData[key]}')

            onnxmltools.utils.save_model(onnx_model, f)
            logging.info(f'{prefix} export success✅, saved as {f}')

            if opt.include_nms and not opt.end2end:
                logging.info(
                    f'{prefix} Registering NMS plugin for ONNX TRT...')
                from utils.add_nms import RegisterNMS

                mo = RegisterNMS(logger=logging,
                                 onnx_model_path=f,
                                 precision='fp16' if img.dtype == torch.float16 else 'fp32', prefix=prefix)
                mo.register_nms(score_thresh=opt.conf_thres, nms_thresh=opt.iou_thres, detections_per_img=opt.topk_all)
                mo.save(f, onnx_MetaData=onnx_MetaData)
                logging.info(f'{prefix} registering NMS plugin for ONNX success✅ {f}')
            filenames.append(f)

        if openVINO:
            prefix = colorstr('OpenVINO:')
            try:
                meta = {'stride': int(max(model.stride)),
                        'names': model.names}
                from tools.auxexport import export_openvino

                logging.info(f'{prefix} Starting export...')
                outputpath, _ = export_openvino(
                    file_=weight, metadata=meta, half=True, prefix=prefix)
                logging.info(
                    f'{prefix} export success✅, saved as: {outputpath}')
                filenames.append(outputpath)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if saved_Model:
            prefix = colorstr('TensorFlow SavedModel:')
            from tools.auxexport import export_saved_model

            outputpath, s_models = export_saved_model(model,
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
            logging.info(f'{prefix} export success✅, saved as {outputpath}')
            filenames.append(outputpath)
        if graphDef:
            prefix = colorstr('TensorFlow GraphDef:')
            try:
                from tools.auxexport import export_pb

                outputpath = export_pb(s_models, weight, prefix=prefix)[0]
                logging.info(
                    f'{prefix} export success✅, saved as {outputpath}')
                filenames.append(outputpath)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if tensorFlowjs:
            prefix = colorstr('TensorFlow.js:')
            try:
                from tools.auxexport import export_tfjs

                outputpath = export_tfjs(file_=weight,
                                         names=labels,
                                         prefix=prefix)[0]
                logging.info(
                    f'{prefix} export success✅, saved as {outputpath}')
                filenames.append(outputpath)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if tensorFlowLite:
            prefix = colorstr('Tensorflow lite:')
            try:
                from tools.auxexport import export_tflite, add_tflite_metadata

                outputpath = export_tflite(s_models, img, weight,
                                           int8=opt.int8,
                                           data=opt.data, nms=opt.nms,
                                           agnostic_nms=opt.agnostic_nms,
                                           stride=gs,
                                           prefix=prefix)[0]
                logging.info(f'{prefix} export success✅, saved as {outputpath}')
                filenames.append(outputpath)
                metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
                logging.info(f'{prefix} adding metadata...')
                add_tflite_metadata(outputpath, metadata=metadata, num_outputs=len(s_models.outputs))
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')

        if len(filenames):
            print('\n')
            prefix = colorstr('Export:')
            for i in filenames:
                logging.info(f'{prefix} {i} is exported.')
            logging.info(
                f'\n{prefix} complete (%.2fs). Visualize with https://netron.app/.' % (time.time() - t))
