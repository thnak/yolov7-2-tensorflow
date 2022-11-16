import cv2 as cv
import os
import math

def lineYOLOtoArray(string=""):
    arr = string.split(" ")
    leng = len(arr)
    for i in range(leng):
        arr[i] = arr[i].replace("\n", "")
    return arr


def read_txtFile_as_arr(path=''):
    arr0 = []
    with open(path, 'r') as file:
        for count, line in enumerate(file):
            arr = lineYOLOtoArray(line)
            arr0.append(arr)
        file.close()
    return arr0

def mathAlpha(img,x='0.0',y='0.0', alpha=0):
    h,w,c = img.shape
    O = [0.5,0.5]
    M = [float(x),float(y)]
    xs = 0
    ys = 0
    _s = 0
    if M[0] > O[0]:
        xs = M[0] - O[0]
    else:
        xs = O[0] - M[0]
        
    if M[1] > O[1]:
        ys = M[1] - O[1]
    else:
        ys = O[1] - M[1]    
    _s = math.sqrt(xs**2 + ys**2)
    # space between M and O
    def xcor(r,alpha):
        
        pass
        
    pass
    

def CloneLabeledValue(imgPath="", new_txt_file="", rotate=0,isFlip=False):
    new_txt_file_exten = []
    file_ = []
    if os.path.exists(imgPath):
        fileName = os.path.basename(imgPath)
        file_ = os.path.splitext(fileName)
    else:
        file_ = ['', '']
    fileName = os.path.basename(new_txt_file)
    new_txt_file_exten = os.path.splitext(fileName)
    print(file_, new_txt_file_exten)
    if file_[0] != '' and file_[1] != '' and new_txt_file_exten[0] != 'null' and new_txt_file_exten[1] == '.txt':
        txtFileRoot = file_[0]+'.txt'
        print('Label root is: '+txtFileRoot)
        if os.path.exists(txtFileRoot):
            print('processing txt file')
            content_arr = read_txtFile_as_arr(txtFileRoot)
            numberOfRow = len(content_arr)
            numberOfColumn = len(content_arr[0])
            if rotate == 0 and isFlip == False:
                print(numberOfColumn, numberOfRow)
                content = ''
                for x in range(numberOfRow):
                    for y in range(numberOfColumn):
                        content = content+content_arr[x][y]
                        if y == numberOfColumn - 1:
                            content = content+'\n'
                        else:
                            content = content+' '
            else:
                rootcordinate = 0.5
                
                pass
                
            if os.path.exists(new_txt_file):
                f = open(new_txt_file, 'w')
            else:
                f = open(new_txt_file, 'x')
            f.write(content)
            f.close()
            print('copy file finshed')
        else:
            print('please label this img')
    else:
        print('image file does not exists or this image does not labeled or the image/txt file path was error')


def CloneLabelWithFliporOrigin(imgPath, isLeft=True, isUp=True):
    pass


filePath = "Screenshot_20221106_094108.png"
string = '0 0.413378 0.233058 0.195679 0.208264' + \
    '\n'+'0 0.413378 0.233058 0.195679 0.208264'

# print(read_txtFile_as_arr(filePath))
# CloneLabeledValue(filePath,'dbas.txt')