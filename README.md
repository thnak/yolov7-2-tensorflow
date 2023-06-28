
<div align="left"></div>
<h1>YOLOv7 supported for tfjs, tflite, saved model, openvino, onnx, rknn.</br>Support for WINDOWS and LINUX</h1>
<h1>Train</h1>
    <p>python train.py --batch 4 --epochs 10 --data 'data/mydataset.yaml' --cache-images disk  --cfg 'cfg/yolov7-tiny.yaml'  --device dml:0 --img-size 640 --weight yolov7.pt</p>
dml:0 to set dml device (windows), cuda:0 to set cuda device, cache-images for faster training disk/ram/no
<h1>Export</h1>
    <p>python export.py --weights "yolov7.pt" --include tfjs/onnx/saved_model/openvino/tflite</p>
    <p>for more information</p>
    <p>python export.py --help</p>
<h1>Usage</h1>
    <p>tfjs model can be used with <a href="https://www.makesense.ai/">makesense.ai</a></p>
    <p>🔥Actions -> Run AI locally -> YOLOv5 object detection using rectangles -> and import all files from *_web_model folder</p>
    <img src='https://user-images.githubusercontent.com/117495750/221329302-c649af5c-f12d-41df-a23c-6dc998e3f90d.png' title='https://www.makesense.ai/'></img>
    </hr>
    <p>also you can follow <a href='https://github.com/Hyuto/yolov5-tfjs'>this step</a> to use (from https://github.com/Hyuto/yolov5-tfjs)</p>
    <img src='https://user-images.githubusercontent.com/117495750/221328795-be9773bc-e070-445f-ac23-22b702c701a8.png' title="https://github.com/Hyuto/yolov5-tfjs"></img>
    </hr>
    <p>.onnx can be used with python.exe detect.py --weights "yolov7-w6.onnx" --nosave --source 0 --view-img 1 (use 0 to freeze screen)</p>
    <img src="https://user-images.githubusercontent.com/117495750/221544577-a65d1b7b-0361-49a4-894c-c8def9dd1a55.png" title='onnx inference'></img>




<details><summary> <h>References</h> </summary>
<ul>
    <li><a href="https://github.com/WongKinYiu/yolov7">Official YOLOv7</a></li>
    <li><a href="https://github.com/meituan/YOLOv6">Official YOLOv6</a></li>
    <li><a href="https://github.com/ultralytics/yolov5">Official YOLOv5</a></li>
    <li><a href="https://github.com/WongKinYiu/yolor">Official YOLOR</a></li>
    <li><a href="https://github.com/Hyuto/yolov5-tfjs">Object Detection using YOLOv5 and Tensorflow.js</a></li>
    <li><a href="https://github.com/Linaom1214/TensorRT-For-YOLO-Series">YOLOv7 with TensorRuntime</a></li>
    <li><a href="https://github.com/SkalskiP/make-sense">Make Sense</a></li>
</ul>
</details>
