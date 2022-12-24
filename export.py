import argparse
import sys
import time
import warnings
import logging
import subprocess
import os
import yaml
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size, check_requirements, colorstr
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolor-csp-c.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx (/end2end/EfficientNMS_TRT)')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--onnx-opset', type=int, default=17, help='onnx opset version')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx(EfficientNMS_TRT)')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    set_logging()
    t = time.time()

    print(f'# Load PyTorch model')
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    modelss = model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
            elif isinstance(m, models.yolo.Detect) or isinstance(m, models.yolo.IDetect):
                m.dynamic = opt.dynamic
            elif isinstance(m, models.yolo.IKeypoint) or isinstance(m, models.yolo.IAuxDetect) or isinstance(m, models.yolo.IBin):
                m.dynamic = opt.dynamic
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)

    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None
    filenames = [None] *10
    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts.save(f)
        print('TorchScript export success✅, saved as %s' % f)
        filenames[0] = f
    except Exception as e:
        print(f'TorchScript export failure🐛🪲: {e}')

    # CoreML export
    try:
        prefix = colorstr('CoreML:')
        import coremltools as ct
        print(f'\n{prefix}Starting CoreML export with coremltools %s...' % ct.__version__)
        ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
        bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
        if bits < 32:
            if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                    ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
            else:
                print(f'{prefix} quantization only supported on macOS, skipping...')

        f = opt.weights.replace('.pt', '.mlmodel')  # filename
        ct_model.save(f)
        print(f'{prefix} CoreML export success✅, saved as %s' % f)
        filenames[1] = f
    except Exception as e:
        print(f'{prefix} CoreML export failure🐛🪲: {e}')
                     
    prefix = colorstr('TorchScript-Lite:')
    try:
        print(f'\n{prefix} Starting TorchScript-Lite export with torch %s...' % torch.__version__)
        f = opt.weights.replace('.pt', '.torchscript.ptl')  # filename
        tsl = torch.jit.trace(model, img, strict=False)
        tsl = optimize_for_mobile(tsl)
        tsl._save_for_lite_interpreter(f)
        print(f'{prefix} TorchScript-Lite export success✅, saved as %s' % f)
        filenames[2] = f
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')

    prefix = colorstr('ONNX:')
    try:
        import onnx
        print(f'\n{prefix} Starting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.eval()
        output_names = ['classes', 'boxes'] if y is None else ['output']
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
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
            else:
                output_axes = {
                    'output': {0: 'batch'},
                }
            dynamic_axes.update(output_axes)
        if opt.grid:
            if opt.end2end:
                print(f'\n{prefix}Starting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
                model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
                if opt.end2end and opt.max_wh is None:
                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True

        torch.onnx.export(model, img, f, verbose=False, opset_version=opt.onnx_opset, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        if opt.end2end and opt.max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        if opt.simplify:
            try:
                import onnxsim
                print(f'{prefix}Starting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'{prefix} Simplifier failure🐛🪲: {e}')
        onnx.save(onnx_model,f)
        print(f'{prefix} export success✅, saved as {f}')

        if opt.include_nms:
            print(f'{prefix} Registering NMS plugin for ONNX...')
            mo = RegisterNMS(f)
            mo.register_nms()
            mo.save(f)
            print(f'{prefix} registering NMS plugin for ONNX success✅ {f}')
        filenames[3] = f
        
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
    prefix = colorstr('Quantize ONNX Models:')
    try:
        saveas = filenames[3].replace('.onnx','_quantize_dynamic.onnx')
        from onnxruntime.quantization import quantize_dynamic, QuantType, quantize, QuantizationMode, StaticQuantConfig, quantize_static, CalibrationDataReader
        quantize_static(filenames[3], filenames[3].replace('.onnx', '_quantize_static.onnx'), weight_type=QuantType.QUInt8)
        quantize_dynamic(filenames[3], saveas,weight_type=QuantType.QUInt8, reduce_range=True)
        print(f'{prefix} export success✅, saved as {saveas}')
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
    
    meta = {'stride': int(max(modelss.stride)), 'names': modelss.names}
    prefix = colorstr('OpenVINO:')
    try:
        from tools.auxexport import export_openvino
        filenames[4], _ = export_openvino(file_=opt.weights,metadata=meta, half=True,prefix=prefix)
        print(f'{prefix} export success✅, saved as {filenames[4]}')
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
        
    prefix = colorstr('TensorFlow SavedModel:')
    try:
        from tools.auxexport import export_saved_model
        filenames[5], s_models = export_saved_model(modelss.cpu(),
                                           img,
                                           opt.weights,
                                           False,
                                           tf_nms=False or False or True,
                                           agnostic_nms=False or True,
                                           topk_per_class=opt.topk_all,
                                           topk_all=opt.topk_all,
                                           iou_thres=opt.iou_thres,
                                           conf_thres=opt.conf_thres,
                                           keras=False,prefix=prefix)
        print(f'{prefix} export success✅, saved as {filenames[5]}')
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
        
    prefix = colorstr('TensorFlow GraphDef:')
    try:
        from tools.auxexport import export_pb
        filenames[6], _ = export_pb(s_models,opt.weights, prefix=prefix)
        print(f'{prefix} export success✅, saved as {filenames[6]}')
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
        
    prefix = colorstr('TensorFlow.js:')
    try:
        from tools.auxexport import export_tfjs
        filenames[7], _ = export_tfjs(file_=opt.weights, prefix=prefix)
        print(f'{prefix} {filenames[2]} is finished')
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
        
        
    prefix = colorstr('Tensorflow lite')
    try:
        import tensorflow as tf
        f = opt.weights.replace('.pt','.pb')
        fo = opt.weights.replace('.pt', '.tflite')
        converter = tf.lite.TFLiteConverter.from_saved_model(f'{f}')
        tf_lite = converter.convert()
        with open(fo, 'wb') as fi:
            fi.write(tf_lite)
        print(f'{prefix} export finished: {e}')
    except Exception as e:
        print(f'{prefix} export failure🐛🪲: {e}')
        
    prefix = colorstr('Export:')
    for i in filenames:
        if i is not None:
            print(f'{prefix} {i} is exported.')
    print(f'\n{prefix} complete (%.2fs). Visualize with https://netron.app/.' % (time.time() - t))