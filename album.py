import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    h,w,c = img.shape
    x,y,witdh,height = bbox
    
    x_min = x - witdh/2
    y_min = y - height/2
    
    x_max = x_min + witdh
    y_max = y_min + height
        
    
    x_min = x_min*w
    x_max = x_max*w
    y_min = y_min*h
    y_max = y_max*h
    
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
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    
image = cv2.imread('app/bird-4887736.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


bboxes = [[0.705413, 0.444593, 0.440589,0.692107]]
category_ids = [17]
category_id_to_name = {17: 'cat'}


transform = A.Compose(
    [
     A.RandomBrightnessContrast(),
     A.RandomGamma(),
     A.RandomScale(),
     A.RandomSunFlare(),
     A.Rotate(limit=(180,-180)),
     A.HueSaturationValue()
     ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']),
)
for _ in range(10):
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    visualize(transformed['image'],transformed['bboxes'], transformed['category_ids'],category_id_to_name,)
    if cv2.waitKey(0) == 27:
        break
