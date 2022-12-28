# example.py
from rembg.bg import remove
import numpy as np
import io
from PIL import Image
import cv2
# Uncomment the following lines if working with trucated image formats (ex. JPEG / JPG)
# In my case I do give JPEG images as input, so i'll leave it uncommented

input_image = "D:/Users/Downloads/pigdataset/train/dataset/images/Img_1039_arg.jpg"
output_image = 'inference/Img_99_arg_DataArgumentation_index_904_0.jpg'


f = cv2.imread(input_image)

result = remove(f)
print(f'{type(result)}, shape {result.shape}')
cv2.imshow('de', result)
cv2.waitKey(0)