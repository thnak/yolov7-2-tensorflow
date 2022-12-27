import numpy as np
import cv2
import os
import time
import yaml

class TensorRT_Engine(object):
    """Torch-TensorRT
        Using for TensorRT inference
    """
    def __init__(self, engine_path, dataset=''):
        """_summary_

        Args:
            engine_path (_type_): _description_
            dataset (str, optional): _description_. Defaults to ''.
            imgsz (tuple, optional): _description_. Defaults to (640,640).
        """
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        import yaml
        import cv2
        import os
        import time
        self.cuda = cuda
        self.trt = trt
        self.Colorselector = BackgroundForegroundColors()
        self.yaml = yaml
        self.cv2 = cv2
        self.os = os
        self.time = time
        
        self.mean = None
        self.std = None

        logger = self.trt.Logger(self.trt.Logger.WARNING)
        logger.min_severity  = self.trt.Logger.Severity.ERROR
        runtime = self.trt.Runtime(logger)
        self.trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins        
        try:
            if os.path.exists(dataset):
                with open(dataset,'r') as dataset_cls_name:
                    data_ = self.yaml.load(dataset_cls_name, Loader=self.yaml.SafeLoader)
                    self.n_classes = data_['nc']
                    self.class_names = data_['names']
            else:
                self.n_classes = 999
                self.class_names = [i for i in range(self.n_classes)]
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()
        except IOError:
            print(f'Error: {IOError}, the item is required')
            exit()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = self.cuda.Stream()
        for binding in engine:
            size = self.trt.volume(engine.get_binding_shape(binding))
            dtype = self.trt.nptype(engine.get_binding_dtype(binding))
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            self.cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(bindings=self.bindings,stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]
        return data
    
    def detect_video(self, video_path, video_outputPath='', conf=0.5, end2end=False, noSave=True):
        """detect objection from video"""
        video_outputPath = self.os.path.join(video_outputPath,'results2.avi')
        
        if not self.os.path.exists(video_path):
            print('video not found, exiting')
            exit()
        cap = self.cv2.VideoCapture(video_path)
        fps = int(round(cap.get(self.cv2.CAP_PROP_FPS)))
        width = int(cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        
        if not noSave:
            print(f'Save video at: {video_outputPath}')
            ffmpeg = FFMPEG_recorder(video_outputPath.replace('.avi','.mp4'),(width, height), fps)
        fps = 0
        avg = []
        timeStart = self.time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = self.preproc(frame, self.imgsz, self.mean, self.std)
            t1 = self.time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (self.time.time() - t1))) / 2
            avg.append(fps)
            frame = self.cv2.putText(frame, "FPS:%d " %fps, (0, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            t2 = self.time.time()
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)
            print(f'FPS: {round(fps,3)}, '+f'nms: {round(self.time.time() - t2,3)}' if end2end else 'postprocess:'+f' {round(self.time.time() - t2,3)}')
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
                frame = self.vis(frame, final_boxes, final_scores, final_cls_inds,conf=conf, class_names=self.class_names)
            if not noSave:
                ffmpeg.writeFrame(frame)
        if not noSave:
            ffmpeg.stopRecorder()
        cap.release()
        
        print(f'Finished! '+f'save at {video_outputPath} ' if not noSave else ''+f'total {round(self.time.time() - timeStart, 2)} second, avg FPS: {round(sum(avg)/len(avg),3)}')

    def inference(self, img_path, conf=0.5, end2end=False):
        """ detect single image
            Return: image
        """
        origin_img = self.cv2.imread(img_path)
        img, ratio = self.preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions=predictions, ratio=ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            origin_img = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,conf=conf, class_names=self.class_names)
        return origin_img

    @staticmethod
    def postprocess(self,predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets
    
    def get_fps(self):
        """Warming up and calculate fps"""
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        t1 = self.time.perf_counter()
        avgT = []
        for _ in range(20):
            _ = self.infer(img)
            t1 = self.time.perf_counter() - t1
            avgT.append(t1)
        print(f'Warming up with {(sum(avgT)/len(avgT)/10)}FPS (etc)')


    def nms(self,boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep


    def multiclass_nms(self,boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy"""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)


    def preproc(self,image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = self.cv2.resize(img,(int(img.shape[1] * r), int(img.shape[0] * r)),interpolation=self.cv2.INTER_LINEAR,).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(round(box[0],3))
            y0 = int(round(box[1],3))
            x1 = int(round(box[2],3))
            y1 = int(round(box[3],3))
            text = '{}:{:.2f}'.format(class_names[cls_id], score)
            txt_color,txt_bk_color = self.Colorselector.getval(cls_id)
            txt_size = self.cv2.getTextSize(text, self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            self.cv2.rectangle(img, (x0, y0), (x1, y1), txt_bk_color, 2)
            c1, c2 = (x0, y0), (x1, y1)
            c2 = c1[0] + txt_size[0], c1[1] - txt_size[1] - 3
            self.cv2.drawContours(img, [np.array([(c1[0] + txt_size[0], c1[1] - txt_size[1] - 3), (c1[0] + txt_size[0], c1[1] ), (c1[0] + txt_size[0] + txt_size[1] + 3, c1[1])])], 0, txt_bk_color, -1, 16)
            self.cv2.rectangle(img, c1, c2, txt_bk_color, -1, self.cv2.LINE_AA)  # filled
            self.cv2.putText(img, text, (c1[0], c1[1] - 2), 0, 0.4, txt_color, thickness=1, lineType=self.cv2.LINE_AA)
        return img


class BackgroundForegroundColors():
    def __init__(self,hyp=None):
        self.COLOR = np.array(
    [   0.850, 0.325, 0.098,
        0.000, 1.000, 0.000,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0]).astype(np.float32).reshape(-1, 3)
        self.textColor = None
        self.bkColor = None
    def getval(self, index=0):
        """get text color, bbox color from index"""
        self.bkColor = (self.COLOR[index] * 255).astype(np.uint8).tolist()
        self.textColor = (0, 0, 0) if np.mean(self.COLOR[index]) > 0.5 else (255, 255, 255)
        return self.textColor, self.bkColor
    def len(self):
        return len(self.COLOR)


import os
import torch
import platform
import subprocess
import shlex

try:
    from pyadl import *
except:
    pass


def getGPUtype():
    try:
        adv = ADLManager.getInstance().getDevices()
        ac = []
        for a in adv:
            ab = [str(a.adapterIndex), str(a.adapterName)]
            ac.append(ab)
    except:
        ac = None
    return ac


class FFMPEG_recorder():
    """Hardware Acceleration for video recording using FFMPEG"""

    def __init__(self, savePath=None, videoDimensions=(1280, 720), fps=30):
        """_FFMPEG recorder_
        Args:
            savePath (__str__, optional): _description_. Defaults to None.
            codec (_str_, optional): _description_. Defaults to None.
            videoDimensions (tuple, optional): _description_. Defaults to (720,1280).
            fps (int, optional): _description_. Defaults to 30FPS.
        """
        self.savePath = savePath
        self.codec = None
        self.videoDementions = videoDimensions
        self.fps = fps
        mySys = platform.uname()
        osType = mySys.system
        if torch.cuda.is_available():
            self.codec = 'hevc_nvenc'
        elif osType == 'Windows' and 'AMD' in str(getGPUtype()):
            self.codec = 'hevc_amf'
        elif osType == 'Linux' and 'AMD' in str(getGPUtype()):
            self.codec = 'hevc_vaapi'
        else:
            self.codec = 'libx264'
        print(f'Using video codec: {self.codec}, os: {osType}')

        self.process = subprocess.Popen(shlex.split(f'ffmpeg -y -s {self.videoDementions[0]}x{self.videoDementions[1]} -pixel_format bgr24 -f rawvideo -r {self.fps} -i pipe: -vcodec {self.codec} -pix_fmt yuv420p -crf 24')+[self.savePath], stdin=subprocess.PIPE)
    def writeFrame(self, image=None):
        """Write frame by frame to video

        Args:
            image (_image_, require): the image will write to video
        """

        self.process.stdin.write(image.tobytes())

    def stopRecorder(self):
        """Stop record video"""
        self.process.stdin.close()
        self.process.wait()
        self.process.terminate()
