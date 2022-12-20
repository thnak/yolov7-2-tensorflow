import argparse
import time
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from termcolor import colored
from models.experimental import attempt_load
from models.yolo import TensorRT_Engine
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, check_git_status, BackgroundForegroundColors
from utils.plots import plot_one_box_with_return
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os
from utils.ffmpeg_ import  FFMPEG_recorder

def detect(opt=None):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    BFC = BackgroundForegroundColors(hyp='./mydataset.yaml')
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model=model, device=device,
                            img_size=imgsz, saveTrace=False)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path = None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    avgTime = [[], []]
    if opt.datacollection:
        save_txt = True
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            twrm = time.time()
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
            print(f'{time.time() - twrm:0.3f} warm up finished')
        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        t4 = time.time()
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                
            imOrigin = im0.copy()
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            os.makedirs(str(save_dir / 'labels'), exist_ok=True)
            txt_path = os.path.join(save_dir, 'labels', p.stem) if dataset.mode == 'image' else os.path.join(save_dir, 'labels', p.stem+f'_{frame}')

            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        f = open(os.path.join(txt_path+'.txt'), 'a')
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        f.close()
                    if save_img or view_img > -1:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        textColor, bboxColor = BFC.getval(index=int(cls))
                        im0 = plot_one_box_with_return(xyxy, im0, label=label, txtColor=textColor, bboxColor=bboxColor, line_thickness=1)

            # Print time (inference + NMS)
            tmInf, tmNms = round(1E3 * (t2 - t1), 3), round(1E3 * (t3 - t2), 3)
            avgTime[0].append(tmInf)
            avgTime[1].append(tmNms)
            print(f'{s}Done. ({tmInf}ms) Inference, ({tmNms}ms) NMS')

            # Stream results
            if view_img > -1:
                cv2.namedWindow(f'{dataset.mode} {path}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'{dataset.mode} {path}', im0)
                if cv2.waitKey(view_img) == 27:
                    break

            # Save results (image with detections)
            if save_img or opt.datacollection:
                if opt.datacollection:
                    im0 = imOrigin
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f"The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        ffmpeg = FFMPEG_recorder(savePath=save_path,videoDimensions=(w,h),fps=fps)
                    ffmpeg.writeFrame(im0)
        print(f'pre-processing {(time.time() - t4):0.3f}s')
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    cv2.destroyAllWindows()
    if (save_img or opt.datacollection) and dataset.mode != 'image':
        ffmpeg.stopRecorder()
        
    print(f'Done. ({time.time() - t0:.3f}s), avgInference {round(sum(avgTime[0])/len(avgTime[0]),3)}ms, avgNMS {round(sum(avgTime[1])/len(avgTime[1]),3)}ms')

def detectTensorRT(tensorrtEngine,opt=None):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    
    device = select_device('cpu')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    del model
    
    
    pred = TensorRT_Engine(engine_path=tensorrtEngine, names=names, imgsz=opt.img_size, confThres=opt.conf_thres, iouThres=opt.iou_thres)
    for path, img, im0s, vid_cap in dataset:
        img = pred.inference(im0s, end2end=True)
        if view_img > -1:
            cv2.namedWindow('TensortRT Engine', img)
            cv2.imshow('TensortRT Engine', cv2.WINDOW_NORMAL)
        if save_img:
            cv2.imwrite(f'{path}trt.jpg',img)
    del pred
    if view_img > -1:
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,default=['./models/yolov7.pt'], help='model.pt path(s)')
    parser.add_argument('--source', type=str,default='inference/images/', help='source file/folder, 0 for webcam')
    parser.add_argument('--img-size', type=int, default=640,help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',type=int, default=-1,help='display results')
    parser.add_argument('--save-txt', action='store_true',help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',help='do not save images/videos')
    parser.add_argument('--datacollection',action='store_true', help='save image and labels')
    parser.add_argument('--classes', nargs='+', type=int,help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true',help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--no-check', action='store_true', help='don`t check requirements')
    opt = parser.parse_args()
    if not opt.no_check:
        check_requirements()
        check_git_status()
    print(f'len{len(opt.weights)}')
    for _ in opt.weights:
        file_extention = os.path.splitext(_)[1]
        if file_extention not in ['.trt', '.engine']:
            with torch.no_grad():
                # update all models (to fix SourceChangeWarning)
                if opt.update:
                    detect(opt=opt)
                    strip_optimizer(f=_,halfModel=True)
                else:
                    detect(opt)
        else:
            detectTensorRT(_,opt)
