# Data augmentation with on click
# Developed by NguyenVanThanh


import time
import cv2
import numpy as np
import os
import tools.SaveImg as SaveImg

inputPath = "D:/Users/Downloads/Imgg/"
outputPath = "D:/Users/Downloads/DataAugmentation"


i = 0
start0 = time.time()
for path in os.listdir(inputPath):
    if os.path.isfile(os.path.join(inputPath, path)):
        i = i + 1
        name = str(i)
        print(path+"\nis processing, image "+name+" is processing with love........................................................................")
        if path is not None:
            path = inputPath+path
            img = cv2.imread(path)
            if img is not None:
                start = time.time()
                SaveImg.ImgGen(img,outputPath,name)
                print(f'Cost {round(time.time()-start,2)}s for this image')
        

def getValueAfterPoint(val):
    num = val.find(".")
    leng = len(val)
    string = ""
    a = num + 1
    for i in range(leng):
        string = string + val[a]
        a = a +1
        if a >= leng:
            break
    return string

print(f'Process completed {round(time.time()-start0,2)}s')
cv2.waitKey(0)
cv2.destroyAllWindows()
