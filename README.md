
<div align="left">
<h1>Train</h1>
    <p>python train.py --batch 4 --epochs 10 --data 'data/mydataset.yaml' --cache-images disk  --cfg 'cfg/yolov7-tiny.yaml'  --device dml:0 --img-size 640 --weight models/best.pt</p>
dml:0 to set dml device (windows), cuda:0 to set cuda device, cache-images for faster training disk/ram/no
<h1>Export</h1>
    <p>python.exe export.py --weights "yolov7.pt" --include tfjs/onnx/saved_model/openvino/tflite</p>
<h1>Usage</h1>
    <p>tfjs can be use with <a href="https://www.makesense.ai/">makesense.ai</a></p>
    <p>.onnx can be use with python.exe detect.py --weights "yolov7-w6.onnx" --nosave --source 0 --view-img 1</p>


<p>References</p>
<details><summary> <b>Expand</b> </summary>
<ul>
    <li><a href="https://github.com/WongKinYiu/yolov7">Official YoLov7</a></li>
    <li><a href="https://github.com/meituan/YOLOv6">Official YoLov6</a></li>
    <li><a href="https://github.com/ultralytics/yolov5">Official YoLov5</a></li>
    <li><a href="https://github.com/WongKinYiu/yolor">Official YoLor</a></li>
    <li><a href="https://github.com/Hyuto/yolov5-tfjs">Object Detection using YOLOv5 and Tensorflow.js</a></li>
    <li><a href="https://blog.csdn.net/qq_56591814/article/details/127172215?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-127172215-blog-115369068.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-127172215-blog-115369068.pc_relevant_default&utm_relevant_index=6">Phân tích siêu tham số</a></li>
    <li><a href="https://qiita.com/omiita/items/bfbba775597624056987">Sự ra đời và giải thích về hàm kích hoạt FReLU</a></li>
    <li><a href="https://www.scirp.org/journal/paperinformation.aspx?paperid=114024">https://www.scirp.org/journal/paperinformation.aspx?paperid=114024</a></li>
    <li><a href="https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine#12">No name</a></li>
    <li><a href="https://blog.tensorflow.org/2021/01/custom-object-detection-in-browser.html?_gl=1*nskhnx*_ga*MTYxODU4MzAzMS4xNjY2NTIwMzc2*_ga_W0YLR4190T*MTY2OTI2Mzk4NC4zLjEuMTY2OTI2Mzk5Ny4wLjAuMA..">Custom object detection in the browser using TensorFlow.js</a></li>
    <li><a href="https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/By_example/Detect_WebGL">Detect WebGL</a></li>
    <li><a href="https://pureadmin.qub.ac.uk/ws/portalfiles/portal/258394935/Deep.pdf">Automated Individual Pig Localisation Tracking and Behaviour Metric Extraction Using Deep Learning</a></li>
    <li><a href='https://medium.com/augmented-startups/how-hyperparameters-of-yolov5-works-ec4d25f311a2'>How do Hyperparameters of YOLOv5 Work?❓</a></li>
    <li><a href="https://da2so.tistory.com/">da2so</a></li>
    <li><a href="https://medium.com/@jalajagr/mean-average-precision-map-explained-in-object-detection-fb61adf67ef4">Mean Average Precision (mAP) Explained in Object Detection</a></li>

</ul>
</details>
