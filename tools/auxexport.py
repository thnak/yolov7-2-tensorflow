import json
import sys
import warnings

from models.experimental import End2End
from utils.general import check_requirements, check_version, colorstr, ONNX_OPSET, ONNX_OPSET_TARGET
import os
import subprocess
import yaml
from pathlib import Path
import re
import torch
import platform

MACOS = platform.system() == 'Darwin'  # macOS environment


def yaml_save(file='data.yaml', data=None):
    # Single-line safe yaml saving
    if data is None:
        data = {}
    with open(file, 'w') as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def export_openvino(file_, metadata, half, prefix='OpenVINO:'):
    check_requirements('openvino-dev')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.inference_engine as ie
    file = Path(file_)
    print(f'\n{prefix} starting export with openvino {ie.__version__}...')
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')

    cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
    subprocess.run(cmd.split(), check=True, env=os.environ)  # export
    save_dir = Path(f) / file.with_suffix('.yaml').name
    yaml_save(save_dir.as_posix(), metadata)  # add metadata.yaml
    return f, None


def export_tfjs(file_, names, prefix='TensorFlow.js:'):
    check_requirements('tensorflowjs')
    import tensorflowjs as tfjs
    print(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
    file = Path(file_)
    f_web = str(file).replace('.pt', '_web_model')  # js dir
    f_pb = str(file).replace('.pt', '.pb')  # *.pb path
    f_json = f'{f_web}/model.json'  # *.json path
    f_labels = f'{f_web}/labels.txt'

    cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
          f'--output_node_names=Identity,Identity_1,Identity_2,Identity_3 {f_pb} {f_web}'
    subprocess.run(cmd.split())

    json = Path(f_json).read_text()
    with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
        subst = re.sub(r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
                       r'"Identity.?.?": {"name": "Identity.?.?"}, '
                       r'"Identity.?.?": {"name": "Identity.?.?"}, '
                       r'"Identity.?.?": {"name": "Identity.?.?"}}}',
                       r'{"outputs": {"Identity": {"name": "Identity"}, '
                       r'"Identity_1": {"name": "Identity_1"}, '
                       r'"Identity_2": {"name": "Identity_2"}, '
                       r'"Identity_3": {"name": "Identity_3"}}}', json)
        j.write(subst)
    with open(f_labels, 'w') as f:
        labels_ = ''
        for x in names:
            labels_ += f'{x}\n'
        labels_ = labels_[:-1]
        f.writelines(labels_)
    return f_web


def export_pb(keras_model, file, prefix='TensorFlow GraphDef:'):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    file = Path(file)
    f = file.with_suffix('.pb')

    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
    return f, None


def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix='TensorFlow SavedModel:'):
    file = Path(file)
    check_requirements(f"tensorflow")
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from models.tf import TFModel

    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    tf_model = TFModel(cfg=model.yaml, model=model.cpu(), nc=len(model.names), imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), name="image", batch_size=None if dynamic else batch_size)
    outputs = tf_model(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()

    if keras:
        keras_model.save(f, save_format='tf')
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(tfm,
                            f,
                            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False) if check_version(
                                tf.__version__, '2.6') else tf.saved_model.SaveOptions())
    return f, keras_model


def export_tflite(keras_model, im, file, int8, data=None, nms=False, agnostic_nms=False, stride=32,
                  prefix='TensorFlow Lite:'):
    """YOLOv5 TensorFlow Lite export"""
    check_requirements(f"tensorflow")
    import tensorflow as tf

    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace('.pt', '-fp16.tflite')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from utils.datasets import LoadImages
        from utils.general import check_dataset, check_yaml
        from models.tf import representative_dataset_gen
        from yaml.loader import SafeLoader
        assert data, f'quantization need data for calibration'
        if check_yaml(data):
            with open(data, 'r') as yamldata:
                data = yaml.load(yamldata, Loader=SafeLoader)
        else:
            raise f'--data must be a yaml file'
        dataset = LoadImages(data['train'], img_size=imgsz, auto=False, stride=stride)
        converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        f = str(file).replace('.pt', '-int8.tflite')
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    return f, None


def add_tflite_metadata(file, metadata, num_outputs):
    """Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata"""
    import contextlib
    with contextlib.suppress(ImportError):
        check_requirements('tflite-support')
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path('/tmp/meta.txt')

        with open(tmp_file, 'w') as meta_f:
            meta_f.write(json.dumps(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        input_meta = _metadata_fb.TensorMetadataT()
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "output"
        output_meta.description = "Tensor[batch, n_pred, [xhwh, cls, cnf]]"
        input_meta.name = "images"
        input_meta.description = "Image to predict"
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]
        model_meta.name = "YOLOv7"
        model_meta.description = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors\n" \
                                 "and this model was exported from https://github.com/thnak/yolov7-2-tensorflow"
        model_meta.license = "https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md"

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([tmp_file.as_posix()])
        populator.populate()
        tmp_file.unlink()


def TryExport_ONNX(weight: Path, model, feed: torch.Tensor, map_device, logging,
                   MetaData: dict = {}, start_points_2_break: list = [], end_points_2_break: list = [],
                   prefix: str = colorstr('ONNX:'), **kwargs):
    check_requirements(('onnx', 'onnxmltools'))
    import onnx
    import onnxmltools

    logging.info(f'\n{prefix} Starting ONNX export with onnx {onnx.__version__}')
    f = weight.as_posix().replace('.pt', '.onnx')  # filename
    output_names = ['classes', 'boxes'] if kwargs["include_nms"] else ['output']

    dynamic_axes = None
    if kwargs["dynamic"]:
        dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                        'output': {0: 'batch', 2: 'y', 3: 'x'}}
    if kwargs["dynamic_batch"]:
        kwargs.batch_size = 'batch'
        dynamic_axes = {
            'images': {
                0: 'batch',
            }, }
        if kwargs["end2end"] and not kwargs["max_hw"]:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'}, }
        else:
            output_axes = {'output': {0: 'batch'}, }
        dynamic_axes.update(output_axes)
    if not kwargs["include_nms"]:
        if kwargs["end2end"]:
            x = 'TensorRT' if not kwargs["max_hw"] else 'ONNXRUNTIME'
            logging.info(f'{prefix} Starting export end2end model for {colorstr(x)}')
            model = End2End(model, kwargs["topk_all"], kwargs["iou_thres"],
                            kwargs["conf_thres"], max(model.input_shape) if kwargs["max_hw"] else None, map_device,
                            len(model.names))
            if kwargs["end2end"] and not kwargs["max_hw"]:
                output_names = ['num_dets', 'det_boxes',
                                'det_scores', 'det_classes']
                shapes = [kwargs["batch_size"], 1, kwargs["batch_size"], kwargs["topk_all"], 4,
                          kwargs["batch_size"], kwargs["topk_all"], kwargs["batch_size"], kwargs["topk_all"]]
            else:
                output_names = ['output']
        else:
            model.model[-1].concat = True
    if kwargs["onnx_opset"] not in ONNX_OPSET:
        logging.info(f'{prefix} onnx opset must be in {ONNX_OPSET}, switching to 12')
        kwargs["onnx_opset"] = ONNX_OPSET_TARGET
    dml = False
    try:
        import torch_directml
        dml = True
    except ImportError:
        pass

    if kwargs["onnx_opset"] not in ONNX_OPSET_TARGET and dml:
        logging.warn(
            f'{prefix} onnx opset tested for version {ONNX_OPSET_TARGET}, '
            f'newer version may have poor performance for ONNXRUNTIME in DmlExecutionProvider')
    torch.onnx.disable_log()
    if feed.dtype != torch.float16 and kwargs["trace"]:
        model = torch.jit.trace(model, feed).eval()
        logging.info(f'{prefix} Traced model!')
    torch.onnx.export(model,
                      feed, f, verbose=kwargs["v"],
                      opset_version=kwargs["onnx_opset"],
                      input_names=['images'],
                      output_names=output_names,
                      training=torch.onnx.TrainingMode.EVAL,
                      dynamic_axes=dynamic_axes,
                      keep_initializers_as_inputs=True)

    if sum([len(start_points_2_break), len(end_points_2_break)]) > 1:
        onnx.utils.extract_model(input_path=f, output_path=f, input_names=start_points_2_break,
                                 output_names=end_points_2_break)
    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    if kwargs["end2end"] and not kwargs["max_hw"]:
        for i in onnx_model.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

    if kwargs["simplify"]:
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

    for index, key in enumerate(MetaData):
        metadata = onnx_model.metadata_props.add()
        metadata.key = key
        metadata.value = str(MetaData[key])

    onnxmltools.utils.save_model(onnx_model, f)

    if kwargs["include_nms"] and not kwargs["end2end"]:
        logging.info(f'{prefix} Registering NMS plugin for ONNX TRT...')
        from utils.add_nms import RegisterNMS

        mo = RegisterNMS(logger=logging,
                         onnx_model_path=f,
                         precision='fp16' if feed.dtype == torch.float16 else 'fp32', prefix=prefix)
        mo.register_nms(score_thresh=kwargs["conf_thres"], nms_thresh=kwargs["iou_thres"],
                        detections_per_img=kwargs["topk_all"])
        mo.save(f, onnx_MetaData=MetaData)
        logging.info(f'{prefix} registering NMS plugin for ONNX success✅ {f}')
    return f


def TryExportTorchScript(weight: Path, model, feed: torch.Tensor, logging,
                         MetaData: dict = {}, lite: bool = False,
                         prefix: str = colorstr('ONNX:')):
    from torch.utils.mobile_optimizer import optimize_for_mobile

    logging.info(
        f'\n{prefix} Starting {"TorchScript-Lite" if lite else "TorchScript"} export with torch {torch.__version__}')
    f = weight.as_posix().replace('.pt', '.torchscript.ptl' if lite else ".torchscript.pt")  # filename
    tsl = torch.jit.trace(model, feed, strict=False)
    MetaData["input_shape"] = [*feed.shape]

    if lite:
        tsl = optimize_for_mobile(tsl)
        tsl._save_for_lite_interpreter(f, _extra_files={"config.txt": json.dumps(MetaData)})
    else:
        torch.jit.save(tsl, f, _extra_files={"config.txt": json.dumps(MetaData)})
    return f


def TryExportCoreML(weight: Path, model, feed: torch.Tensor, logging,
                    prefix: str = colorstr('ONNX:'), **kwargs):
    check_requirements('coremltools')
    import coremltools as ct

    logging.info(f'\n{prefix} Starting CoreML export with coremltools {ct.__version__}')
    if kwargs["end2end"]:
        from models.yolo import iOSModel

        ts = iOSModel(model, feed)
        ts = torch.jit.trace(ts, feed, strict=False)
    ct_model = ct.convert(ts,
                          inputs=[ct.ImageType('image',
                                               shape=feed.shape,
                                               scale=1 / 255.0,
                                               bias=[0, 0, 0])])
    bits, mode = (8, 'kmeans_lut') if kwargs['int8'] else (
        16, 'linear') if kwargs['fp16'] else (32, None)
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
    return f
