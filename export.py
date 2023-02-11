import datetime
from utils.add_nms import RegisterNMS
from utils.torch_utils import select_device
from utils.general import set_logging, check_img_size, check_requirements, colorstr
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
sys.path.append('./')  # to run '$ python *.py' files in subdirectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,default=['./best.pt'], help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true',help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true',help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--include', nargs='+', type=str,default='', help='export format')
    parser.add_argument('--end2end', action='store_true',help='export end2end onnx (/end2end/EfficientNMS_TRT)')
    parser.add_argument('--max-hw', type=int, default=None,help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100,help='topk objects for every images')
    parser.add_argument('--iou-thres', '-iou', type=float,default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', '-conf', type=float,default=0.25, help='conf threshold for NMS')
    parser.add_argument('--onnx-opset', type=int, default=17,help='onnx opset version, 11 for DmlExecutionProvider')
    parser.add_argument('--device', default='cpu',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true',help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true',help='registering EfficientNMS_TRT plugin to export TensorRT engine')
    parser.add_argument('--fp16', action='store_true',help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true',help='CoreML INT8 quantization')
    parser.add_argument('--v', action='store_true', help='Verbose log')
    parser.add_argument('--author', type=str,default='Nguyễn Văn Thạnh', help="author's name")
    opt = parser.parse_args()
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    set_logging()
    logging.info(f'\n{opt}\n')
    
    opt.include = [x.lower() for x in opt.include] if isinstance(
        opt.include, list) else [opt.include.lower()]

    torchScript = any(x in ['torchscript'] for x in opt.include)
    torchScriptLite = any(x in ['torchscriptlite'] for x in opt.include)
    ONNX = any(x in ['onnx', 'open', 'openvino'] for x in opt.include)
    openVINO = any(x in ['openvino', 'open'] for x in opt.include)
    tensorFlowjs = any(x in ['tfjs'] for x in opt.include)
    tensorFlowLite = any(x in ['tflite'] for x in opt.include)
    coreML = any(x in ['coreml'] for x in opt.include)
    saved_Model = any(x in ['saved_model', 'tfjs', 'tflite']
                      for x in opt.include)
    graphDef = any(x in ['saved_model', 'grapdef', 'tfjs', 'tflite']
                   for x in opt.include)

    t = time.time()
    opt.weights = [os.path.realpath(x) for x in opt.weights] if isinstance(
        opt.weights, (tuple, list)) else [os.path.realpath(opt.weights)]
    for weight in opt.weights:
        logging.info(f'# Load PyTorch model')
        device, gitstatus = select_device(opt.device)
        map_device = 'cpu' if device.type == 'privateuseone' else device
        model = attempt_load(weight, map_location=map_device)  # load FP32 model
        ckpt = torch.load(weight, map_location=map_device)

        best_fitness = ckpt['best_fitness'].tolist() if 'best_fitness' in ckpt else 'unknown'
        best_fitness = str(best_fitness[0]) if isinstance(best_fitness, (tuple, list)) else str(best_fitness)
        
        epoch = ckpt['epoch'] if 'epoch' in ckpt else 'unknown'
        training_results = ckpt['training_results'] if 'training_results' in ckpt else 'unknown'
        ema = ckpt['ema'] if 'ema' in ckpt else 'unknown'
        updates = ckpt['updates'] if 'updates' in ckpt else 'unknown'
        optimizer = ckpt['optimizer'] if 'optimizer' in ckpt else 'unknown'
        wandb_id = ckpt['wandb_id'] if 'wandb_id' in ckpt else 'unknown'
        hyp = ckpt['hyp'] if 'hyp' in ckpt else 'unknown'
        trainOpt = ckpt['opt'] if 'opt' in ckpt else 'unknown'

        labels = model.names
        model_ori = model
        model_Gflop = model.info()
        gs = int(max(model.stride))  # grid size (max stride)

        input_shape = ckpt['input shape'] if 'input shape' in ckpt else [3,640,640]
        img = torch.zeros(opt.batch_size, *input_shape).to(map_device)
        model.eval()
        if device.type in ['cuda', 'dml'] and opt.fp16:
            img, model = img.half(), model.half()
            logging.info(
                f'Export with shape {input_shape}, FP16, best fitness: {best_fitness}')
        else:
            logging.info(
                f'Export with shape {input_shape}, FP32, best fitness: {best_fitness}')
        # Update model
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
                elif isinstance(m, (models.yolo.Detect, models.yolo.IDetect, models.yolo.IKeypoint, models.yolo.IAuxDetect, models.yolo.IBin)):
                    m.dynamic = opt.dynamic
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)

        model.model[-1].export = False  # set Detect() layer grid export
        y = model(img)  # dry run
        # model output shape
        shape = tuple((y[0] if isinstance(y, tuple) else y).shape)

        if opt.include_nms:
            model.model[-1].include_nms = True
            y = None
        filenames = []
        # TorchScript export
        if torchScript:
            try:
                prefix = colorstr('TorchScript:')
                logging.info(
                    f'\n{prefix} Starting TorchScript export with torch %s...{torch.__version__}')
                f = weight.replace('.pt', '.torchscript.pt')  # filename
                ts = torch.jit.trace(model, img, strict=False)
                ts.save(f)
                logging.info(f'{prefix} export success✅, saved as {f}')
                filenames.append(f)
            except Exception as e:
                logging.info(f'{prefix} export failure❌: {e}')
            # CoreML export
        if coreML:
            try:
                prefix = colorstr('CoreML:')
                import coremltools as ct
                logging.info(
                    f'\n{prefix}Starting CoreML export with coremltools {ct.__version__}')
                ct_model = ct.convert(ts, inputs=[ct.ImageType(
                    'image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
                bits, mode = (8, 'kmeans_lut') if opt.int8 else (
                    16, 'linear') if opt.fp16 else (32, None)
                if bits < 32:
                    if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                        with warnings.catch_warnings():
                            # suppress numpy==1.20 float warning
                            warnings.filterwarnings(
                                "ignore", category=DeprecationWarning)
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
                logging.info(f'{prefix} export failure❌: {e}')
        if ONNX:
            prefix = colorstr('ONNX:')
            import onnx
            import onnxmltools
            logging.info(
                f'\n{prefix} Starting ONNX export with onnx {onnx.__version__}')
            f = weight.replace('.pt', '.onnx')  # filename
            model.eval()
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
                if opt.end2end and opt.max_hw is None:
                    output_axes = {
                        'num_dets': {0: 'batch'},
                        'det_boxes': {0: 'batch'},
                        'det_scores': {0: 'batch'},
                        'det_classes': {0: 'batch'}, }
                else:
                    output_axes = {'output': {0: 'batch'}, }
                dynamic_axes.update(output_axes)
            if opt.end2end:
                x = 'TensorRT' if opt.max_hw is None else 'ONNXRUNTIME'
                logging.info(
                    f'{prefix} Starting export end2end onnx model for {colorstr(x)}')
                model = End2End(model, opt.topk_all, opt.iou_thres,
                                opt.conf_thres, opt.max_hw, map_device, len(labels))
                if opt.end2end and opt.max_hw is None:
                    output_names = ['num_dets', 'det_boxes',
                                    'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True
            if opt.onnx_opset < 11 and opt.onnx_opset > 17:
                opt.onnx_opset = 11
            if opt.onnx_opset > 11:
                logging.info(
                    f'{prefix} onnx opset tested for version 11, newer version may have poor performance for ONNXRUNTIME in DmlExecutionProvider')
            torch.onnx.export(model, img, f, verbose=opt.v,
                              opset_version=opt.onnx_opset,
                              input_names=['images'],
                              output_names=output_names, training=torch.onnx.TrainingMode.EVAL,
                              dynamic_axes=dynamic_axes)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            if opt.end2end and opt.max_hw is None:
                for i in onnx_model.graph.output:
                    for j in i.type.tensor_type.shape.dim:
                        j.dim_param = str(shapes.pop(0))

            if opt.simplify:
                try:
                    import onnxsim
                    logging.info(f'{prefix} Starting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    logging.info(f'{prefix} Simplifier failure❌: {e}')

            onnx.save(onnx_model, f=f)

            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            logging.info(f'{prefix} writing metadata for model...')
            onnx_MetaData = {'export gitstatus': gitstatus,
                             'traing gitstatus': str(ckpt['gitstatus']) if 'gitstatus' in ckpt else 'unknown',
                             'best fitness': best_fitness,
                             'epoch': str(ckpt['epoch']) if 'epoch' in ckpt else 'unknown',
                             'training result': str(ckpt['training_results']) if 'training_results' in ckpt else 'unknown',
                             'ema': str(ckpt['ema']) if 'ema' in ckpt else 'unknown',
                             'updates': str(ckpt['updates']) if 'updates' in ckpt else 'unknown',
                             'optimizer': str(ckpt['optimizer']) if 'optimizer' in ckpt else 'unknown',
                             'wandb id':  str(ckpt['wandb_id']) if 'wandb_id' in ckpt else 'unknown',
                             'hyp': str(ckpt['hyp']) if 'hyp' in ckpt else 'unknown',
                             'training opt': str(ckpt['opt']) if 'opt' in ckpt else 'unknown',
                             'opset version': str(opt.onnx_opset),
                             'stride': str(gs),
                             'nc': str(len(labels)),
                             'names': str(labels), 'ort-nms': 'True' if opt.end2end and opt.max_hw else 'False',
                             'export date': datetime.datetime.now().isoformat('#'),
                             'train date': str(ckpt['date']) if 'date' in ckpt else 'unknown',
                             'author': str(opt.author),
                             'exporting opt': str(opt),
                             'pytorch model info': str(model_Gflop)
                             }
            for index, key in enumerate(onnx_MetaData):
                metadata = onnx_model.metadata_props.add()
                metadata.key = key
                metadata.value = onnx_MetaData[key]
                logging.info(f'key: {key}, value: {onnx_MetaData[key]}')

            onnxmltools.utils.save_model(onnx_model, f)
            logging.info(f'{prefix} export success✅, saved as {f}')

            if opt.include_nms:
                logging.info(
                    f'{prefix} Registering NMS plugin for ONNX...')
                mo = RegisterNMS(f)
                mo.register_nms()
                mo.save(f)
                logging.info(
                    f'{prefix} registering NMS plugin for ONNX success✅ {f}')
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
                logging.info(f'{prefix} export failure❌: {e}')

        if saved_Model:
            prefix = colorstr('TensorFlow SavedModel:')
            from tools.auxexport import export_saved_model
            outputpath, s_models = export_saved_model(ckpt,
                                                      img,
                                                      weight,
                                                      False,
                                                      tf_nms=True,
                                                      agnostic_nms=True,
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
                outputpath, _ = export_pb(s_models, weight, prefix=prefix)
                logging.info(
                    f'{prefix} export success✅, saved as {outputpath}')
                filenames.append(outputpath)
            except Exception as e:
                logging.info(f'{prefix} export failure❌: {e}')
        if tensorFlowjs:
            prefix = colorstr('TensorFlow.js:')
            try:
                from tools.auxexport import export_tfjs
                outputpath, _ = export_tfjs(
                    file_=weight, names=labels, prefix=prefix)
                logging.info(
                    f'{prefix} export success✅, saved as {outputpath}')
                filenames.append(outputpath)
            except Exception as e:
                logging.info(f'{prefix} export failure❌: {e}')

        if tensorFlowLite:
            prefix = colorstr('Tensorflow lite:')
            try:
                import tensorflow as tf
                f = weight.replace('.pt', '.pb')
                fo = weight.replace('.pt', '.tflite')
                if os.path.exists(fo):

                    converter = tf.lite.TFLiteConverter.from_saved_model(
                        f'{f}')
                    tf_lite = converter.convert()
                    if tf_lite:
                        with open(fo, 'wb') as fi:
                            fi.write(tf_lite)
                        filenames.append(fo)
                        logging.info(
                            f'{prefix} export finished, save as:  {fo}')
                    else:
                        logging.info(f'{prefix} export failure❌: {fo}')
            except Exception as e:
                logging.info(f'{prefix} export failure❌: {e}')
        if len(filenames):
            print('\n')
            prefix = colorstr('Export:')
            for i in filenames:
                logging.info(f'{prefix} {i} is exported.')
            logging.info(
                f'\n{prefix} complete (%.2fs). Visualize with https://netron.app/.' % (time.time() - t))
