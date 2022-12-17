from utils.utils import TensorRT_Engine
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-mydataset", "--mydataset", help="mydataset yaml file to get class name")
    parser.add_argument("-o", "--output",default='./', help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("--end2end", action="store_true", help="use end2end engine")
    parser.add_argument("--nosave", action="store_true", help="use end2end engine")
    parser.add_argument("--view", action="store_true", help="use end2end engine")
    args = parser.parse_args()
    print(args)
    if not args.engine or not args.mydataset:
      pred = TensorRT_Engine(engine_path=args.engine)
      pred.get_fps()
      img_path = args.image
      video = args.video
      if img_path:
        origin_img = pred.inference(img_path, conf=0.1, end2end=args.end2end)
        if not args.nosave:
          cv2.imwrite("%s" %args.output , origin_img)
        if args.view:
          cv2.namedWindow('TensorRT viewer', cv2.normalize)
          cv2.imshow('TensorRT viewer', origin_img)
          if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
      if video:
        pred.detect_video(video, conf=0.1, end2end=args.end2end, video_outputPath=args.output,noSave=args.nosave) # set 0 use a webcam
    else:
      print(f'The engine file path, dataset file path is required')