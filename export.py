import argparse
import sys
import time
import warnings
import logging
import subprocess
import os
import yaml
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from termcolor import colored
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size, check_requirements, colorstr
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS
import datetime



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',nargs='+', type=str, default=['./best.pt'], help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size [height, width]')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx (/end2end/EfficientNMS_TRT)')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--onnx-opset', type=int, default=17, help='onnx opset version, 11 for DmlExecutionProvider')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='registering EfficientNMS_TRT plugin to export TensorRT engine')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')
    parser.add_argument('--v', action='store_true', help='Verbose log')
    parser.add_argument('--author', type=str, default='Nguyễn Văn Thạnh', help="author's name")
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    set_logging()
    logging.info(f'\n{opt}\n')
    
    
    t = time.time()
    opt.weights = [os.path.realpath(x) for x in opt.weights] if isinstance(opt.weights, (tuple, list)) else [os.path.realpath(opt.weights)]
    for weight in opt.weights:
        logging.info(f'# Load PyTorch model')
        device, gitstatus = select_device(opt.device)
        model = attempt_load(weight, map_location=device)  # load FP32 model
        labels = model.names
        model_ori = model
        model_Gflop = model.info()

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
        filenames = []
        # TorchScript export
        try:
            prefix = colorstr('TorchScript:')
            logging.info(f'\n{prefix} Starting TorchScript export with torch %s...{torch.__version__}' )
            f = weight.replace('.pt', '.torchscript.pt')  # filename
            ts = torch.jit.trace(model, img, strict=False)
            ts.save(f)
            logging.info(f'{prefix} export success✅, saved as {f}')
            filenames[0] = f
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')

        # CoreML export
        try:
            prefix = colorstr('CoreML:')
            import coremltools as ct
            logging.info(f'\n{prefix}Starting CoreML export with coremltools {ct.__version__}')
            ct_model = ct.convert(ts, inputs=[ct.ImageType('image', shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])])
            bits, mode = (8, 'kmeans_lut') if opt.int8 else (16, 'linear') if opt.fp16 else (32, None)
            if bits < 32:
                if sys.platform.lower() == 'darwin':  # quantization only supported on macOS
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress numpy==1.20 float warning
                        ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
                else:
                    logging.info(f'{prefix} quantization only supported on macOS, skipping...')
            f = weight.replace('.pt', '.mlmodel')  # filename
            ct_model.save(f)
            logging.info(f'{prefix} CoreML export success✅, saved as %s' % f)
            filenames.append(f)
        except Exception as e:
            logging.info(f'{prefix} CoreML export failure🐛🪲: {e}')
                        
        prefix = colorstr('TorchScript-Lite:')
        try:
            logging.info(f'\n{prefix} Starting TorchScript-Lite export with torch {torch.__version__}' )
            f = weight.replace('.pt', '.torchscript.ptl')  # filename
            tsl = torch.jit.trace(model, img, strict=False)
            tsl = optimize_for_mobile(tsl)
            tsl._save_for_lite_interpreter(f)
            logging.info(f'{prefix} TorchScript-Lite export success✅, saved as {f}')
            filenames.append(f)
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')

        prefix = colorstr('ONNX:')
        try:
            import onnx
            import onnxmltools
            logging.info(f'\n{prefix} Starting ONNX export with onnx {onnx.__version__}')
            f = weight.replace('.pt', '.onnx')  # filename
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
                        'det_classes': {0: 'batch'},}
                else:
                    output_axes = {'output': {0: 'batch'},}
                dynamic_axes.update(output_axes)
            if opt.grid:
                if opt.end2end:
                    x = 'TensorRT' if opt.max_wh is None else 'ONNXRUNTIME'
                    logging.info(f'\n{prefix} Starting export end2end onnx model for {colorstr(x)}\n')
                    model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
                    if opt.end2end and opt.max_wh is None:
                        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                        shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                                opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                    else:
                        output_names = ['output']
                else:
                    model.model[-1].concat = True
            if opt.onnx_opset < 11 and opt.onnx_opset > 17:
                opt.onnx_opset = 11
            if opt.onnx_opset > 11:
                logging.info(f'{prefix} onnx opset tested for version 11, newer version may have poor performance for ONNXRUNTIME in DmlExecutionProvider')
            torch.onnx.export(model, img, f, verbose=opt.v,
                              opset_version=opt.onnx_opset,
                              input_names=['images'],
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
                    logging.info(f'{prefix} Starting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    logging.info(f'{prefix} Simplifier failure🐛🪲: {e}')
                    
            onnx.save(onnx_model,f=f)
            
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model     
                              
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'gitstatus'
            metadata.value = gitstatus
             
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'opset version'
            metadata.value = str(opt.onnx_opset)
            
            metadata = onnx_model.metadata_props.add()      
            metadata.key = 'stride'
            metadata.value = str(gs)
            
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'nc'
            metadata.value = str(len(labels))
            
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'names'
            metadata.value = str(labels)

            metadata = onnx_model.metadata_props.add()
            metadata.key = 'ort-nms'
            metadata.value = 'True' if opt.end2end and opt.max_wh else 'False'            
            
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'date'
            metadata.value = str(datetime.datetime.now())
            
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'author'
            metadata.value = opt.author
            
            metadata = onnx_model.metadata_props.add()
            metadata.key = 'optional export'
            metadata.value = str(opt)
            
            metadata = onnx_model.metadata_props.add()
            metadata.key = "pytorch model info"
            metadata.value = str(model_Gflop)
            
            logging.info(f'\n{prefix} metadata: {onnx_model.metadata_props}\n')
            onnxmltools.utils.save_model(onnx_model,f)
            logging.info(f'{prefix} export success✅, saved as {f}')

            if opt.include_nms:
                logging.info(f'{prefix} Registering NMS plugin for ONNX...')
                mo = RegisterNMS(f)
                mo.register_nms()
                mo.save(f)
                logging.info(f'{prefix} registering NMS plugin for ONNX success✅ {f}')
            filenames.append(f)
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
            
            
        prefix = colorstr('Quantize ONNX Models:')
        try:
            saveas = weight.replace('.pt', '.onnx').replace('.onnx','_quantize_dynamic.onnx')
            from onnxruntime.quantization import quantize_dynamic, QuantType, quantize, QuantizationMode, StaticQuantConfig, quantize_static, CalibrationDataReader
            # quantize_static(weight.replace('.pt', '.onnx'), filenames[3].replace('.onnx', '_quantize_static.onnx'), weight_type=QuantType.QUInt8)
            quantize_dynamic(weight.replace('.pt', '.onnx'), saveas,weight_type=QuantType.QUInt8, reduce_range=True)
            logging.info(f'{prefix} export success✅, saved as: {saveas}')
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
        
        
        prefix = colorstr('OpenVINO:')
        try:
            meta = {'stride': int(max(model_ori.stride)), 'names': model_ori.names}
            from tools.auxexport import export_openvino
            logging.info(f'{prefix} Starting export...')
            outputpath, _ = export_openvino(file_=weight,metadata=meta, half=True,prefix=prefix)
            logging.info(f'{prefix} export success✅, saved as: {outputpath}')
            filenames.append(outputpath)
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
            
        prefix = colorstr('Pytorch2TensorFlow:')
        try:
            import onnx_tf
            from tools.pytorch2tensorflow import _onnx_to_tf, _tf_to_tflite
            logging.info(f'{prefix} Starting export with...{onnx_tf.__version__}')
            outputpath = weight.replace('.pt', '01.pb') 
            _onnx_to_tf(onnx_model_path=weight.replace('.pt', '.onnx'), output_path=outputpath)
            logging.info(f'{prefix} export success: {outputpath}')
            filenames.append(outputpath)
            inputpath = outputpath
            outputpath = outputpath.replace('.pb', '.tflite') 
            
            _tf_to_tflite(tf_model_path=inputpath, output_path=outputpath)
            logging.info(f'{prefix} export success, saved as: {outputpath}')
            filenames.append(outputpath)
        except Exception as ex:
            logging.info(f'{prefix} export failure: {ex}')
            
        prefix = colorstr('TensorFlow SavedModel:')
        try:
            from tools.auxexport import export_saved_model
            outputpath, s_models = export_saved_model(model.cpu(),
                                            img,
                                            weight,
                                            False,
                                            tf_nms=False or False or True,
                                            agnostic_nms=False or True,
                                            topk_per_class=opt.topk_all,
                                            topk_all=opt.topk_all,
                                            iou_thres=opt.iou_thres,
                                            conf_thres=opt.conf_thres,
                                            keras=False,prefix=prefix)
            logging.info(f'{prefix} export success✅, saved as {outputpath}')
            filenames.append(outputpath)
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
            
        prefix = colorstr('TensorFlow GraphDef:')
        try:
            from tools.auxexport import export_pb
            outputpath, _ = export_pb(s_models,weight, prefix=prefix)
            logging.info(f'{prefix} export success✅, saved as {outputpath}')
            filenames.append(outputpath)
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
            
        prefix = colorstr('TensorFlow.js:')
        try:
            from tools.auxexport import export_tfjs
            outputpath, _ = export_tfjs(file_=weight, prefix=prefix)
            logging.info(f'{prefix} {outputpath} is finished')
            filenames.append(outputpath)
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
            
        prefix = colorstr('Tensorflow lite:')
        try:
            import tensorflow as tf
            f = weight.replace('.pt','.pb')
            fo = weight.replace('.pt', '.tflite')
            if os.path.exists(fo):
                
                converter = tf.lite.TFLiteConverter.from_saved_model(f'{f}')
                tf_lite = converter.convert()
                if tf_lite:
                    with open(fo, 'wb') as fi:
                        fi.write(tf_lite)
                    filenames.append(fo)
                    logging.info(f'{prefix} export finished, save as:  {fo}')
                else:
                    logging.info(f'{prefix} export failure🐛🪲: {fo}')
        except Exception as e:
            logging.info(f'{prefix} export failure🐛🪲: {e}')
        
        print('')
        prefix = colorstr('Export:')
        for i in filenames:
            logging.info(f'{prefix} {i} is exported.')
        logging.info(f'\n{prefix} complete (%.2fs). Visualize with https://netron.app/.' % (time.time() - t))