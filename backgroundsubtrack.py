from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    fgMask = cv.erode(fgMask, kernel, iterations=1) 
    fgMask = cv.dilate(fgMask, kernel, iterations=1)
    fgMask = cv.GaussianBlur(fgMask, (3,3), 0)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
    _,fgMask = cv.threshold(fgMask,130,255,cv.THRESH_BINARY)

    fgMask = cv.Canny(fgMask,20,200)
    contours,_ = cv.findContours(fgMask,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        (x, y, w, h) = cv.boundingRect(contours[i])
        area = cv.contourArea(contours[i])
        if area > 300:
            cv.drawContours(fgMask, contours[i], 0, (0, 0, 255), 6)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)




    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    cv.namedWindow('FG Mask', cv.WINDOW_NORMAL)
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break