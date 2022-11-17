import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
image = cv2.imread('bird-4887736.png')
print(image)
if image:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    visualize(image)
else:
    print('err')