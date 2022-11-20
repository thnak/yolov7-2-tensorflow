import cv2
import os
from termcolor import colored


def resizeImg(imgPath="D:/Users/Downloads/Imgg" , reSize=False, imgOutputPath="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/tmr2"):
    if imgPath.endswith('/'):
        pass
    else:
        imgPath += '/'
        
    if imgOutputPath.endswith('/'):
        pass
    else:
        imgOutputPath += '/'
            
    array_path = os.listdir(imgPath)
    arrImg = []
    for _ in array_path:
        if _.endswith('.jpg') or _.endswith('.png'):
            arrImg.append(imgPath+_)
    print(colored(f'Total {len(arrImg)} image found', 'yellow'))
    count = 0


    for imgP in arrImg:
        img = cv2.imread(imgP)
        h, w = img.shape[:2]
        mpx = h*w
        if reSize:
            if mpx > 1000000:
                height = round(h / 1000,0)
                width = round(w / 1000,0)
                height = round(h / height,0)
                width = round(w / width,0)
                width = int(round(width,0))
                height = int(round(height,0))
                img = cv2.resize(img,(width,height), interpolation = cv2.INTER_AREA)
        namefile = imgOutputPath+'Img_' + str(count)+'_arg.jpg'
        cv2.imwrite(namefile, img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        print(colored(f'Processing {namefile}', 'white'))
        count += 1

resizeImg(reSize=True)
