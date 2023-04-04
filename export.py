import datetime
from utils.torch_utils import select_device, prune
from utils.general import set_logging, check_img_size, check_requirements, colorstr, ONNX_OPSET, ONNX_OPSET_TARGET, \
    gb2mb
from utils.general import set_logging, check_img_size, check_requirements, colorstr
from utils.general import set_logging, check_img_size, check_requirements, colorstr, ONNX_OPSET, ONNX_OPSET_TARGET
from utils.general import set_logging, check_img_size, check_requirements, colorstr, ONNX_OPSET, ONNX_OPSET_TARGET
from utils.activations import Hardswish, SiLU
from models.experimental import attempt_load, End2End
import models
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn
import torch
import argparse
import sys
import time
import warnings
import logging
import os
import numpy as np
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['./best.pt'], help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--include', nargs='+', type=str, default='', help='export format')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx (/end2end/EfficientNMS_TRT)')
    parser.add_argument('--max-hw', '--ort', action='store_true', default=None,
                        help='end2end onnxruntime')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', '-iou', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', '-conf', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--onnx-opset', type=int, default=17, help='onnx opset version, 11 for DmlExecutionProvider')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    t = time.time()
    opt.weights = [os.path.realpath(x) for x in opt.weights] if isinstance(
        opt.weights, (tuple, list)) else [os.path.realpath(opt.weights)]
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    for weight in opt.weights:
        prefix = colorstr('Export:')
        logging.info(f'{prefix} Load PyTorch model')
        device, gitstatus = select_device(opt.device)
        map_device = 'cpu' if device.type == 'privateuseone' else device
        with torch.no_grad():
            ckpt = torch.load(weight, map_location=map_device)
            try:
                model = attempt_load(weight, map_location=map_device).to(map_device).eval()  # load FP32 model
            except Exception:
                model = ckpt['model'].float().eval()
            for m in model.parameters():
                m.requires_grad = False
            ckpt.pop('model', None)
            ckpt.pop('ema', None)
            ckpt.pop('optimizer', None)
            ckpt.pop('updates', None)
            # prune(model)
            model_ori = deepcopy(model)

        best_fitness = model.best_fitness if hasattr(model, 'best_fitness') else 0.
        total_image = model.total_image if hasattr(model, 'total_image') else [0]
        input_shape = model.input_shape if hasattr(model, 'input_shape') else ([3, 640, 640] if model.is_p5() else [3, 1280, 1280])
        model_version = model.model_version if hasattr(model, 'model_version') else 0
        model.best_fitness = best_fitness
        model.model_version = model_version
        model.total_image = total_image
        model.input_shape = input_shape
        labels = model.names
        gs = int(max(model.stride.max(), 32))  # grid size (max stride)

        img = torch.zeros(opt.batch_size, *input_shape, device=map_device)

        # Update model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):  # assign export-friendly activations
                if isinstance(m, (models.yolo.Detect, models.yolo.IDetect, models.yolo.IAuxDetect)):
                    m.dynamic = opt.dynamic

        model_Gflop = model.info(verbose=False, img_size=input_shape)
        logging.info(model_Gflop)
        model.model[-1].export = coreML  # set Detect() layer grid export, for coreml export set to True
        y = model(img)  # dry run

        if device.type in ['cuda'] and opt.fp16:
            img = img.half()
            model = model.half()
        else:
            if opt.fp16:
                logging.warning(f'Export with fp16 only support for CUDA device, yours {device.type}')

        # model output shape
        shape = tuple((y[0] if isinstance(y, (tuple, list)) else y).shape)
        logging.info(f'{prefix} model output shape {shape}')

        if opt.include_nms:
            model.model[-1].include_nms = True
            y = None
        filenames = []
        # TorchScript export
        if torchScript:
            try:
                prefix = colorstr('TorchScript:')
                logging.info(
                    f'\n{prefix} Starting TorchScript export with torch{torch.__version__}')
                f = weight.replace('.pt', '.torchscript.pt')  # filename
                ts = torch.jit.trace(model, img, strict=False)
                ts.save(f)
                logging.info(f'{prefix} export success✅, saved as {f}')
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌:\n{e}')
            # CoreML export
        if coreML:
            try:
                prefix = colorstr('CoreML:')
                check_requirements('coremltools')
                import coremltools as ct

                logging.info(
                    f'\n{prefix} Starting CoreML export with coremltools {ct.__version__}')
                ct_model = ct.convert(ts, inputs=[ct.ImageType(
                    'image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
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
                f = weight.replace('.pt', '.mlmodel')  # filename
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
                f = weight.replace('.pt', '.torchscript.ptl')  # filename
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
            f = weight.replace('.pt', '.onnx')  # filename
            output_names = ['classes',
                            'boxes'] if y is None else ['output']
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
            torch.onnx.export(model,
                              img, f, verbose=opt.v,
                              opset_version=opt.onnx_opset,
                              input_names=['images'],
                              output_names=output_names,
                              training=torch.onnx.TrainingMode.EVAL,
                              dynamic_axes=dynamic_axes,
                              keep_initializers_as_inputs=True)
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

            # onnx.save(onnx_model, f=f)

            # onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            logging.info(f'{prefix} writing metadata for model...')

            onnx_MetaData = {'model_infor': model_Gflop,
                             'export_gitstatus': gitstatus,
                             'best_fitness': best_fitness,
                             'stride': gs,
                             'nc': len(labels),
                             'names': labels,
                             'total_image': total_image,
                             'export_date': datetime.datetime.now().isoformat('#'),
                             'exporting_opt': vars(opt),
                             }
            key_prefix = colorstr('yellow', 'key:')
            for index, key in enumerate(ckpt):
                if key == 'model':
                    continue
                if key == 'best_fitness':
                    ckpt[key] = ckpt[key].tolist()[0] if isinstance(ckpt[key], (np.ndarray, torch.Tensor)) else ckpt[key]
                onnx_MetaData[key] = ckpt[key]

            for index, key in enumerate(onnx_MetaData):
                metadata = onnx_model.metadata_props.add()
                metadata.key = key
                metadata.value = str(onnx_MetaData[key])
                logging.info(f'{key_prefix} {key}, value: {onnx_MetaData[key]}')

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
                meta = {'stride': int(max(model_ori.stride)),
                        'names': model_ori.names}
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

            outputpath, s_models = export_saved_model(model_ori,
                                                      img,
                                                      weight,
                                                      False,
                                                      tf_nms=tensorFlowjs or opt.nms or opt.agnostic_nms,
                                                      agnostic_nms=tensorFlowjs or opt.agnostic_nms,
                                                      topk_per_class=opt.topk_all,
                                                      topk_all=opt.topk_all,
                                                      iou_thres=opt.iou_thres,
                                                      conf_thres=opt.conf_thres,
                                                      keras=False, prefix=prefix)
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
                check_requirements('tensorflowjs')
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
