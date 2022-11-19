import cv2
import albumentations as A
import os

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    h,w,c = img.shape
    x,y,witdh,height = bbox
    x_min = (x - witdh/2)*w
    y_min = (y - height/2)*h
    x_max = (x_min + witdh)*w
    y_max = (y_min + height)*h
    
    x_min, x_max, y_min, y_max = int(x_min), int(x_max ), int(y_min), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
    
def getName_extention(stringPath=''):
    file_ = os.path.basename(stringPath)
    fileName = os.path.splitext(file_)
    return fileName
    
    
def readYoLoBbox(stringPath=''):
    txt = open(stringPath, 'r')
    stringtxt = txt.read()
    arr = stringtxt.split('\n')
    aux= []
    for i in arr:
        a = i.split(' ')
        au = list(map(float,a))
        au[0] = int(au[0])
        aux.append(au)
    return aux
    
def DataArgumentation(imginputPath='', bboxInputPath='', bbboxOutputPath='', imgOutputPath='', cropSize=[416,416], showImg=False):
    if bboxInputPath =='':
        bboxInputPath = imginputPath
        print('Bbox input path is null, using img input path to scan...\n')
    
    print(f'Scaning {imginputPath}...')
    arrImageInputPath_, arrImageInputPath=[], []
    arrImageInputPath_ = os.listdir(imginputPath)
    print(arrImageInputPath_)
    cou = 0
    for imgpath in arrImageInputPath_:
        if imgpath.endswith('.jpg') or imgpath.endswith('.png') or imgpath.endswith('.jpeg') or imgpath.endswith('.PNG') or imgpath.endswith('.JPG'):
            arrImageInputPath.append(imgpath)
            cou+=1
    print(f'Found total {cou} image files')
    
    arrBboxInputPath, arrBboxInputPath_ = [], []
    arrBboxInputPath_ = os.listdir(bboxInputPath)
    cou = 0
    for imgpath in arrBboxInputPath_:
        if os.path.isfile(imgpath) and imgpath.endswith('.txt'):
            arrBboxInputPath.append(imgpath)
            cou +=1
    print(f'Found total {cou} txt files')
    
    img_bbox = []
    for imgpath in arrImageInputPath:
        imgNamed = getName_extention(imgpath)[0]
        for arrBbox in arrBboxInputPath:
            namedBbox = getName_extention(arrBbox)[0]
            if imgNamed in namedBbox or imgNamed == namedBbox:
                ar = [imgpath, arrBbox]
                img_bbox.append(ar)
            else:
                print(f'{imgpath} were not labeled')
    
    print(f'Total {len(img_bbox)} files was labeled, processing...')
    
    
    for img_bbox_ in img_bbox:
        image_, bbox_= img_bbox_[0], img_bbox_[1]
        bbox = readYoLoBbox(bbox_)
        image_ = imginputPath+'/'+image_
        bbox_ = bboxInputPath+'/'+bbox_
        process(imgPath=image_, bboxes=bbox, viewImg=True)
        
def process(imgPath='', bboxes=[], category_ids=[], category_id_to_name={0: 'pig',1:'dog'}, imgIndex=0, viewImg=False,imgOutputPath='', bboxOutputPath=''):
    image = cv2.imread(imgPath)
    print(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    category_ids = category_ids
    for _ in range(len(bboxes)):
        category_ids.append(bboxes[_][0])
        bboxes[_] = [bboxes[_][1], bboxes[_][2], bboxes[_][3],bboxes[_][4]]
    
    
    # bboxes = [[0.704009, 0.580384, 0.466981,0.839232],[0.307193, 0.621161, 0.515330, 0.757678]]

    p = 1/4
    transform = A.Compose(
        [
        A.RandomBrightnessContrast(p=p),
        A.RandomGamma(p=p),
        A.RandomSunFlare(p=p),
        A.HueSaturationValue(p=p, hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50,always_apply=False),
        A.ToGray(p=0.2),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRain(p=p),
        A.ISONoise(p=p),
        A.RandomResizedCrop(p=p,height=416,width=416),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
    )
    count = 0
    print(bboxes)
    print(category_ids)
    for _ in range(40):
        nameImg = imgOutputPath+'index' +str(imgIndex)+str(count)+'.jpg'
        count+=1
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        img = visualize(transformed['image'],transformed['bboxes'], transformed['category_ids'],category_id_to_name)
        print(transformed['bboxes'])
        # cv2.imwrite(nameImg,img,[cv2.IMWRITE_JPEG_QUALITY, 75])
        if viewImg:
            cv2.namedWindow('a',cv2.WINDOW_NORMAL)
            cv2.imshow('a',img)
        if cv2.waitKey(1) == 27:
            break


DataArgumentation(imginputPath='inference')
