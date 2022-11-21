import cv2
import albumentations as A
import os
import sys
from termcolor import colored


def visualize_bbox(img, bbox, class_name, thickness=2):
    h, w = img.shape[:2]
    x, y, witdh, height = bbox

    x_min = (x - witdh/2)*w
    y_min = (y - height/2)*h

    x_max = ((x_min/w) + witdh)*w
    y_max = ((y_min/h) + height)*h
    # print(f'xmax :{x_max} ymax" {y_max} bbox: {bbox} x: {h} y: {w}')
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=(255, 0, 0), thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), (255, 0, 0), -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def getName_extention(stringPath=''):
    file_ = os.path.basename(stringPath)
    fileName = os.path.splitext(file_)
    return fileName


def readYoLoBbox(stringPath=''):
    txt = open(stringPath, 'r')
    stringtxt = txt.read()
    arr = stringtxt.split('\n')
    categoryId = []
    bboxes = []
    for i in arr:
        a = i.split(' ')
        au = list(map(float, a))
        au[0] = int(au[0])
        categoryId.append(au[0])
        bboxes.append([au[1], au[2], au[3], au[4]])

    return bboxes, categoryId


def DataArgumentation(imgInputPath='', bboxInputPath='', bboxOutputPath='', imgOutputPath='', rangeArgumentation=10, cropSize=[416, 416], showImg=False, save=False):
    if imgInputPath == '':
        print(f'Image input path is empty, exiting program')
        exit()
    if imgInputPath.endswith('/'):
        pass
    else:
        imgInputPath = imgInputPath+'/'

    print(colored(f'Scaning folder:','yellow'), colored(f'{imgInputPath}','white'))
    arrImageInputPath_, arrImageInputPath = [], []
    arrImageInputPath_ = os.listdir(imgInputPath)
    cou = 0
    for imgpath in arrImageInputPath_:
        if imgpath.endswith('.jpg') or imgpath.endswith('.png') or imgpath.endswith('.jpeg') or imgpath.endswith('.PNG') or imgpath.endswith('.JPG'):
            if imgInputPath.endswith('/'):
                arrImageInputPath.append(imgInputPath+''+imgpath)
            else:
                arrImageInputPath.append(imgInputPath+'/'+imgpath)
            cou += 1
    print(colored(f'Found total {cou} image files', 'green'))

    if bboxInputPath == '':
        bboxInputPath = imgInputPath
        print('Bbox input path is empty, using image input path to scan')
    else:
        print(colored(f'Scaning folder:','yellow'), colored(f'{bboxInputPath}','white'))

    if bboxInputPath.endswith('/'):
        pass
    else:
        bboxInputPath = bboxInputPath+'/'
    arrBboxInputPath, arrBboxInputPath_ = [], []
    arrBboxInputPath_ = os.listdir(bboxInputPath)
    cou = 0
    for imgpath in arrBboxInputPath_:
        if imgpath.endswith('.txt'):
            if bboxInputPath.endswith('/'):
                arrBboxInputPath.append(bboxInputPath+''+imgpath)
            else:
                arrBboxInputPath.append(bboxInputPath+'/'+imgpath)
            cou += 1
    print(colored(f'Found total {cou} txt files', 'green'))

    img_bbox = []
    for imgpath in arrImageInputPath:
        imgNamed = getName_extention(imgpath)[0]+getName_extention(imgpath)[1]

        txtPath = imgNamed.replace('.jpg', '.txt')
        txtPath = txtPath.replace('.png', '.txt')
        txtPath = bboxInputPath + txtPath
        if os.path.exists(txtPath):
            ar = [imgpath, txtPath]
            img_bbox.append(ar)
        else:
            print(colored(f'{imgpath} were not labeled','red'))
    print(f'Save output images to: {imgOutputPath}')
    print(f'Save output labels to: {bboxOutputPath}')
    print(colored(f'Total {len(img_bbox)} files was labeled, processing with {rangeArgumentation} argumentation...', 'yellow'))
    if imgOutputPath == '':
        imgOutputPath = imgInputPath
    else:
        if imgOutputPath.endswith('/'):
            pass
        else:
            imgOutputPath += '/'
    if bboxOutputPath == '':
        bboxOutputPath = bboxInputPath
    else:
        if bboxOutputPath.endswith('/'):
            pass
        else:
            bboxOutputPath += '/'
    inx = 0
    lengImg_bbox = len(img_bbox)
    for img_bbox_ in img_bbox:
        image_, bbox_ = img_bbox_[0], img_bbox_[1]
        bbox, cate = readYoLoBbox(bbox_)

        sys.stdout.write(('='*int((inx/lengImg_bbox)*100))+(''*(lengImg_bbox-inx)
                                                            )+("\r [ %d" % int((inx/lengImg_bbox)*100)+"% ] "))
        sys.stdout.flush()
        process(rangeArgumentation=rangeArgumentation, 
                imgPath=image_, bboxes=bbox, category_ids=cate,
                imgIndex=inx, viewImg=True, imgOutputPath=imgOutputPath,
                save=save,
                bboxOutputPath=bboxOutputPath)
        inx += 1
    print('Finished')

def process(imgPath='', rangeArgumentation=1, cropSizeRate=[1, 1], viewImg_miliseconds=1, bboxes=[], category_ids=[], category_id_to_name={0: 'person', 1: 'cow', 2:'dairy cow', 3:'buffalo',4:'pig',5:'sheep', 6:'burro',7:'horse',8:'rabbit',9:'deer',10:'goat', 11:'dog', 12:'cat', 13:'chicken', 14:'duck', 15:'mallard', 16:'dove', 17:'googse', 18:'musk duck', 19:'galeeny', 20:'turkey', 21:'eel',22:'cockroach'}, imgIndex=0, viewImg=False, save=False, imgOutputPath='', bboxOutputPath=''):
    image = cv2.imread(imgPath)
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fileName = getName_extention(imgPath)[0]

    p = 1/4
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=p),
            A.RandomGamma(p=p),
            A.HueSaturationValue(p=p, hue_shift_limit=50, sat_shift_limit=50,
                                 val_shift_limit=50, always_apply=False),
            A.ToGray(p=0.2),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.RandomRain(p=p),
            A.ISONoise(p=p),
            A.RandomResizedCrop(p=p, height=h*cropSizeRate[0], width=w*cropSizeRate[1]
                                ) if w > 416 and h > 416 else A.RandomResizedCrop(p=0, height=416, width=416),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
    )
    count = 0
    for _ in range(rangeArgumentation):

        nameImg = imgOutputPath+fileName+'_DataArgumentation_index_' + \
            str(imgIndex)+'_'+str(count)+'.jpg'
        count += 1
        transformed = transform(
            image=image, bboxes=bboxes, category_ids=category_ids)
        stringtxt = ''
        for cate, bbox in zip(transformed['category_ids'], transformed['bboxes']):
            stringtxt += str(cate) + ' '+str(bbox[0])+' '+str(
                bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+'\n'
        if save:
            cv2.imwrite(nameImg, transformed['image'], [
                        cv2.IMWRITE_JPEG_QUALITY, 75])
            nameImg = nameImg.replace(imgOutputPath, bboxOutputPath)
            nameImg = nameImg.replace('.jpg', '.txt')
            # print(f'file name: {nameImg}')
            if os.path.exists(nameImg):
                file_ = open(nameImg, 'w')
                file_.write(stringtxt[:-1])
                file_.close()
            else:
                file_ = open(nameImg, 'x')
                file_.write(stringtxt[:-1])
                file_.close()

        if viewImg:
            img = visualize(transformed['image'], transformed['bboxes'],
                            transformed['category_ids'], category_id_to_name)
            cv2.namedWindow('DataArgumentation', cv2.WINDOW_NORMAL)
            cv2.imshow('DataArgumentation', img)
            if cv2.waitKey(viewImg_miliseconds) == 27:
                break


def reWriteIndex(inputFolder='', outputFolder='', changesCateOldIndex_newIndex=[[0], [4]]):
    if inputFolder.endswith('/'):
        pass
    else:
        inputFolder += '/'
    if outputFolder.endswith('/'):
        pass
    else:
        outputFolder += '/'
    if outputFolder == '':
        outputFolder = inputFolder
    if os.path.isdir(inputFolder) and os.path.isdir(outputFolder):
        arrItem = os.listdir(inputFolder)
        print(colored(f'Found {len(arrItem)} txt files', 'yellow'))
        for item in arrItem:
            if item.endswith('.txt'):
                item = inputFolder+item
                bbox, cate = readYoLoBbox(item)
                for index1 in range(len(cate)):
                    for index2 in range(len(changesCateOldIndex_newIndex[0])):
                        if cate[index1] == changesCateOldIndex_newIndex[0][index2]:
                            cate[index1] = changesCateOldIndex_newIndex[1][index2]
                        else:
                            print('dif')
                stringTxt = ''
                for _ in range(len(cate)):
                    stringTxt += str(cate[_]) +' '+ str(bbox[_][0])+' '+ str(bbox[_][1])+' '+ str(bbox[_][2])+' '+ str(bbox[_][3])+'\n'
                stringTxt = stringTxt[:-1]
                print(f'cate: {cate}, bbox: {bbox}')
                print(stringTxt)
                item = item.replace(inputFolder, outputFolder)
                fileTxt = open(item,'w')
                fileTxt.write(stringTxt)
                fileTxt.close()
    else:
        print(f'inputFolder or outputFolder does not exists')
DataArgumentation(showImg=True,save=True,imgOutputPath="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/newimage",bboxOutputPath="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/newlabel",imgInputPath="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/images", bboxInputPath="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/labels")


# reWriteIndex(inputFolder="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/labels", outputFolder="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/newlabel")
