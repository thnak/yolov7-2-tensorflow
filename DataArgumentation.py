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


def visualize(image, bboxes, category_ids, category_id_to_name, filename=''):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
    
def DataArgumentation(imginputPath='', bboxInputPath='', bbboxOutputPath='', imgOutputPath='', cropSize=[416,416], showImg=False):
    
    arrImageInputPath_, arrImageInputPath=[], []
    arrImageInputPath_ = os.listdir(imginputPath)
    
    cou = 0
    for imgpath in arrImageInputPath_:
        if os.path.isfile(imgpath):
            if imgpath.endswith('.jpg') or imgpath.endswith('.png'):
                arrImageInputPath.append(imgpath)
                cou+=1
    print(f'Total {cou} image files\n')
    
    arrBboxOutputPath, arrBboxOutputPath_ = [], []
    arrBboxOutputPath_ = os.listdir(bboxInputPath)
    cou = 0
    for imgpath in arrBboxOutputPath_:
        if os.path.isfile(imgpath) and imgpath.endswith('.txt'):
            arrBboxOutputPath.append(imgpath)
            cou +=1
    print(f'Total {cou} txt files\n')
    
    for imgpath in arrImageInputPath:
        pass
    
    
    image = cv2.imread('inference/bird-4887736.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bboxes = [[0.704009, 0.580384, 0.466981,0.839232],[0.307193, 0.621161, 0.515330, 0.757678]]
    category_ids = [17,18]
    category_id_to_name = {17: 'cat',18:'dog'}

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
    for _ in range(40):
        name = 'deploy/img_s' +str(count)+'.jpg'
        count+=1
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        img = visualize(transformed['image'],transformed['bboxes'], transformed['category_ids'],category_id_to_name,name)
        cv2.namedWindow('a',cv2.WINDOW_NORMAL)
        cv2.imwrite(name,img,[cv2.IMWRITE_JPEG_QUALITY, 75])
        cv2.imshow('a',img)
        if cv2.waitKey(1) == 27:
            break
