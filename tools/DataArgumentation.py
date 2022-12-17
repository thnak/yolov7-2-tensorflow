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
    txt.close()
    arr = stringtxt.split('\n')
    categoryId = []
    bboxes = []
    for i in arr:
        a = i.split(' ')
        if a != ['']:
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

    print(colored(f'Scaning folder:', 'yellow'),
          colored(f'{imgInputPath}', 'white'))
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
        print(colored(f'Scaning folder:', 'yellow'),
              colored(f'{bboxInputPath}', 'white'))

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
            print(colored(f'{imgpath} were not labeled', 'red'))
    print(f'Save output images to: {imgOutputPath}')
    print(f'Save output labels to: {bboxOutputPath}')
    print(colored(
        f'Total {len(img_bbox)} files was labeled, processing with {rangeArgumentation} argumentation...', 'yellow'))
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
        try:
            bbox, cate = readYoLoBbox(bbox_)
        except Exception as ex:
            print(f'Error: {ex}\nPath: {bbox_}')
            continue
        sys.stdout.write(('='*int((inx/lengImg_bbox)*100))+(''*(lengImg_bbox-inx)
                                                            )+("\r [ %d" % int((inx/lengImg_bbox)*100)+"% ] "))
        sys.stdout.flush()
        process(rangeArgumentation=rangeArgumentation,
                imgPath=image_, bboxes=bbox, category_ids=cate,
                imgIndex=inx, viewImg=showImg, imgOutputPath=imgOutputPath,
                save=save,
                bboxOutputPath=bboxOutputPath)
        inx += 1
    print('Finished')


def process(imgPath='', 
            rangeArgumentation=1,
            cropSizeRate=[1, 1], 
            viewImg_miliseconds=1, bboxes=[], 
            category_ids=[], category_id_to_name={0: 'person', 1: 'cow', 2: 'dairy cow', 3: 'buffalo', 4: 'pig', 5: 'sheep', 6: 'burro', 7: 'horse', 8: 'rabbit', 9: 'deer', 10: 'goat', 11: 'dog', 12: 'cat', 13: 'chicken', 14: 'duck', 15: 'mallard', 16: 'dove', 17: 'googse', 18: 'musk duck', 19: 'galeeny', 20: 'turkey', 21: 'eel', 22: 'cockroach'}, 
            imgIndex=0, viewImg=False, save=False, 
            imgOutputPath='', bboxOutputPath=''):
    
    image = cv2.imread(imgPath)
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fileName = getName_extention(imgPath)[0]

    p = 0
    transform = A.Compose(
        [
            # A.RandomBrightnessContrast(p=p),
            # A.RandomGamma(p=p),
            # A.HueSaturationValue(p=p, hue_shift_limit=50, sat_shift_limit=50,
            #                      val_shift_limit=50, always_apply=False),
            # A.ToGray(p=p),
            # A.HorizontalFlip(p=p),
            # A.VerticalFlip(p=p),
            # A.RandomRain(p=p),
            # A.ISONoise(p=p),
            # A.RandomResizedCrop(p=p, height=h*cropSizeRate[0], width=w*cropSizeRate[1]
            #                     ) if w > 416 and h > 416 else A.RandomResizedCrop(p=0, height=416, width=416),
            # A.ChannelDropout(p=1)
            A.Cutout(p=1,max_h_size=1,max_w_size=1,num_holes=100)
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
    )
    count = 0.1
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
            
            cv2.imwrite(nameImg, cv2.cvtColor(transformed['image'],cv2.COLOR_BGR2RGB), [
                        cv2.IMWRITE_JPEG_QUALITY, 55])
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
                    stringTxt += str(cate[_]) + ' ' + str(bbox[_][0])+' ' + str(
                        bbox[_][1])+' ' + str(bbox[_][2])+' ' + str(bbox[_][3])+'\n'
                stringTxt = stringTxt[:-1]
                print(f'cate: {cate}, bbox: {bbox}')
                print(stringTxt)
                item = item.replace(inputFolder, outputFolder)
                fileTxt = open(item, 'w')
                fileTxt.write(stringTxt)
                fileTxt.close()
    else:
        print(f'inputFolder or outputFolder does not exists')


def splitDataset(inputFolder='', outputFolder='', trainRatio=0.75):
    img_formats = ['.bmp', '.jpg', '.jpeg', '.png',
                   '.tif', '.tiff', '.dng', '.webp', '.mpo']

    yoloArrFiles = [[], []]
    if os.path.isdir(inputFolder) and os.path.isdir(outputFolder):
        imagesInputFolder = os.path.join(inputFolder, 'images')
        labelsInputFolder = os.path.join(inputFolder, 'labels')
        for item in os.listdir(imagesInputFolder):
            for exImage in img_formats:
                if item.endswith(exImage):
                    imageNamed = os.path.join(imagesInputFolder, item)
                    labelNamed = imageNamed.replace(
                        imagesInputFolder, labelsInputFolder)
                    for _ in img_formats:
                        labelNamed = labelNamed.replace(_, '.txt')
                    labelNamed = os.path.join(labelNamed)
                    if os.path.exists(labelNamed) and imageNamed:
                        yoloArrFiles[1].append(labelNamed)
                        yoloArrFiles[0].append(imageNamed)
        print(
            f'total image files: {len(yoloArrFiles[0])}, total label files: {len(yoloArrFiles[1])}')
        if os.listdir(outputFolder):
            print(
                'The output folder is not empty, require empty folder for train, val, test sub-folder inside')
        else:
            valRatio = (1 - trainRatio) * 0.4
            testRatio = (1 - trainRatio) * 0.6
            print(
                f'Starting split folder... {trainRatio*100}% train, {valRatio*100}% val, {testRatio*100}% test.')
            trainFolder = os.path.join(outputFolder, 'train')
            trainFolder_, valFolder_, testFolder_ = [], [], []
            valFolder = os.path.join(outputFolder, 'val')
            testFolder = os.path.join(outputFolder, 'test')
            if not os.path.isdir(trainFolder):
                os.makedirs(trainFolder)
                trainFolder_.append(os.path.join(trainFolder, 'images'))
                trainFolder_.append(os.path.join(trainFolder, 'labels'))
                for _ in trainFolder_:
                    os.makedirs(_)
            else:
                trainFolder_.append(os.path.join(trainFolder, 'images'))
                trainFolder_.append(os.path.join(trainFolder, 'labels'))
            if not os.path.isdir(valFolder):
                os.makedirs(valFolder)
                valFolder_.append(os.path.join(valFolder, 'images'))
                valFolder_.append(os.path.join(valFolder, 'labels'))
                for _ in valFolder_:
                    os.makedirs(_)
            else:
                valFolder_.append(os.path.join(valFolder, 'images'))
                valFolder_.append(os.path.join(valFolder, 'labels'))
            if not os.path.isdir(testFolder):
                os.makedirs(testFolder)
                testFolder_.append(os.path.join(testFolder, 'images'))
                testFolder_.append(os.path.join(testFolder, 'labels'))
                for _ in testFolder_:
                    os.makedirs(_)
            else:
                testFolder_.append(os.path.join(testFolder, 'images'))
                testFolder_.append(os.path.join(testFolder, 'labels'))
            import random
            import shutil
            tempArray = yoloArrFiles
            itempathTrain, itempathVal, itempathTest = [
                [], []], [[], []], [[], []]
            for _ in range(int(round(float(len(tempArray[0]))*valRatio, 0))):
                im = random.randrange(0, len(tempArray[0]) - 1)
                itempathVal[0].append(
                    tempArray[0][im].replace(imagesInputFolder, '')[1:])
                itempathVal[1].append(
                    tempArray[1][im].replace(labelsInputFolder, '')[1:])

                shutil.copyfile(yoloArrFiles[0][im], os.path.join(
                    valFolder_[0], tempArray[0][im].replace(imagesInputFolder, '')[1:]))
                shutil.copyfile(yoloArrFiles[1][im], os.path.join(
                    valFolder_[1], tempArray[1][im].replace(labelsInputFolder, '')[1:]))
                tempArray[0].remove(tempArray[0][im])
                tempArray[1].remove(tempArray[1][im])

            print(f'valFolder: {len(itempathVal[0])} files, finished')

            for _ in range(int(round(float(len(tempArray[0]))*testRatio, 0))):
                im = random.randrange(0, len(tempArray[0]) - 1)
                itempathTest[0].append(
                    tempArray[0][im].replace(imagesInputFolder, '')[1:])
                itempathTest[1].append(
                    tempArray[1][im].replace(labelsInputFolder, '')[1:])
                shutil.copyfile(yoloArrFiles[0][im], os.path.join(
                    testFolder_[0], tempArray[0][im].replace(imagesInputFolder, '')[1:]))
                shutil.copyfile(yoloArrFiles[1][im], os.path.join(
                    testFolder_[1], tempArray[1][im].replace(labelsInputFolder, '')[1:]))
                tempArray[0].remove(tempArray[0][im])
                tempArray[1].remove(tempArray[1][im])

            print(f'testFolder: {len(itempathTest[0])} files, finished')
            itempathTrain = tempArray
            for im in range(len(itempathTrain[0])):
                shutil.copyfile(yoloArrFiles[0][im], os.path.join(
                    trainFolder_[0], itempathTrain[0][im].replace(imagesInputFolder, '')[1:]))
                shutil.copyfile(yoloArrFiles[1][im], os.path.join(trainFolder_[1], itempathTrain[1][im].replace(labelsInputFolder, '')[1:]))
            print(f'trainFolder: {len(itempathTrain[0])} files, finished')


def removecls(path='', targetBbx=[]):
    if path:
        path = os.path.join(path)
        listPath = os.listdir(path)
        for item in listPath:
            itemPath = os.path.join(path, item)
            try:
                bbx, catlo = readYoLoBbox(itemPath)
            except Exception as ex:
                print(f'Error: {ex}\nPath: {itemPath}')
                continue
            stringlabel = ''
            for i in range(len(bbx)):
                if str(catlo[i]) in targetBbx:
                    stringlabel += str(0)+' ' + str(bbx[i][0])+' ' + str(
                        bbx[i][1])+' ' + str(bbx[i][2])+' ' + str(bbx[i][3])+'\n'
            print(stringlabel[:-1])
            print('')
            file_ = open(itemPath, 'w')
            if stringlabel != '':
                file_.write(stringlabel[:-1])
            else:
                file_.write('')
            file_.close()

# removecls(path="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/labels_", targetBbx=['4'])


# splitDataset(inputFolder="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train", outputFolder="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/cattle")
DataArgumentation(showImg=True, save=True, rangeArgumentation=1,
                  imgOutputPath="D:/Users/Downloads/New folder",
                  bboxOutputPath="D:/Users/Downloads/New folder",
                  imgInputPath="D:/Users/Downloads/pigdataset/train/dataset/images",
                  bboxInputPath="D:/Users/Downloads/pigdataset/train/dataset/labels")

def reduce_Quality_of_PNG_image(path='', outpath='',count=0, jpgQuality=50):
    listpath = os.listdir(path)
    for item in listpath:
        itemPath = os.path.join(path,item)
        img = cv2.imread(itemPath)
        savePath = os.path.join(outpath,f'background_{count}.jpg')
        cv2.imwrite(savePath,img,[cv2.IMWRITE_JPEG_QUALITY,jpgQuality])
        count+=1
# reduce_Quality_of_PNG_image("D:/Users/Downloads/New folder", "D:/Users/Downloads/New folder (3)")


# reWriteIndex(inputFolder="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/labels", outputFolder="D:/Users/Downloads/Pig behavior.v1-walking.yolov7pytorch/train/newlabel")


def splitTrain2Val(datasetFolder='', rate = 0.15):
    trainFolder = os.path.join(datasetFolder,'train')
    valFolder = os.path.join(datasetFolder,'val')
    sub_trainFolder, sub_valFolder = [], []
    sub_trainFolder.append(os.path.join(trainFolder,'images'))
    sub_trainFolder.append(os.path.join(trainFolder,'labels'))
    sub_valFolder.append(os.path.join(valFolder,'images'))
    sub_valFolder.append(os.path.join(valFolder,'labels'))
    
    if valFolder == '':
        print('Not empty val folder')
        exit()
    
    data = [[],[]]
    result = [[],[]]
    for item in os.listdir(sub_trainFolder[0]):
        data[0].append(item)
    for item in os.listdir(sub_trainFolder[1]):
        data[1].append(item)
    from pathlib import Path
    
    if len(data[0]) > len(data[1]):
        for i in data[1]:
            strpath = Path(i)
            iNamed = strpath.stem
            for item in data[0]:
                if iNamed in item:
                    result[0].append(item)
                    result[1].append(i)        
    else:
        print(f'total {len(data[0])} item')
        for i in data[0]:
            strpath = Path(i)
            iNamed = strpath.stem
            for item in data[1]:
                if iNamed in item:
                    result[0].append(item)
                    result[1].append(i)  
    print(f'total: {len(result[0])} items')
    totalVal = int(round(len(result[0]) * rate,0))
    totalVal = len(result[0]) - totalVal
    import random
    import shutil
    for _ in range(totalVal):
        inx = random.randint(0,len(result[0])-1)
        result[0].remove(result[0][inx])
        result[1].remove(result[1][inx])
    print(f'len {len(result[0])}')
    for _ in range(len(result[0])):
        imgPath = result[0][_]
        txtPath = result[1][_]
        imgPathout = Path(os.path.join(sub_valFolder[0], imgPath))
        txtPathout = Path(os.path.join(sub_valFolder[1], txtPath))     
        txtPathin = Path(os.path.join(sub_trainFolder[1], txtPath))
        imgPathin = Path(os.path.join(sub_trainFolder[0], imgPath))
        print(f' in: {imgPathin}, out: {imgPathout}') 
        os.makedirs(os.path.join(valFolder, sub_valFolder[0]), exist_ok=True)
        os.makedirs(os.path.join(valFolder, sub_valFolder[1]), exist_ok=True)
        shutil.copyfile(txtPathin, txtPathout, follow_symlinks=False)
        shutil.copyfile(imgPathin,imgPathout,follow_symlinks=False)     
        
# splitTrain2Val("D:/Users/Downloads/dataset/train", "D:/Users/Downloads/dataset/val", 0.2)