import cv2
import os
import tools.filter as filter
import numpy as np


def saveImg(img, name,  outputPath,order):
        errorCount = 0
        completeCout = 0
        err = 0
        named = "Croped_GrayScale_noRotate_noFlip" + name + order
        result = filter.Image2Grayscale(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
    
        named = "Croped_GrayScale_noRotate_Flip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,0,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])    
    
        named = "Croped_GrayScale_Rotate45_noFlip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])      
        
        named = "Croped_GrayScale_Rotate45_Flip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_GrayScale_Rotate90_Flip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_GrayScale_Rotate270_Flip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_GrayScale_Rotate90_noFlip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_GrayScale_Rotate270_noFlip" + name + order
        result = filter.Image2Grayscale(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
      
        
        
            
        named = "Croped_BlueAndRedFilter_noRotate_noFlip" + name + order
        result = filter.BlueAndRedFilter(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_BlueAndRedFilter_noRotate_Flip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_BlueAndRedFilter_Rotate90_Flip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_BlueAndRedFilter_Rotate90_noFlip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_BlueAndRedFilter_Rotate45_Flip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_BlueAndRedFilter_Rotate45_noFlip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_BlueAndRedFilter_Rotate180_Flip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,180,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_BlueAndRedFilter_Rotate180_noFlip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,180,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                     
               
        named = "Croped_BlueAndRedFilter_Rotate270_Flip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
        named = "Croped_BlueAndRedFilter_Rotate270_noFlip" + name + order
        result = filter.BlueAndRedFilter(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
               
            #    
        named = "Croped_redFilter_noRotate_noFlip" + name + order
        result = filter.redFilter(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        
        named = "Croped_redFilter_noRotate_Flip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_redFilter_Rotate90_Flip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_redFilter_Rotate90_noFlip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_redFilter_Rotate45_Flip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_redFilter_Rotate45_noFlip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_redFilter_Rotate180_Flip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,180,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_redFilter_Rotate180_noFlip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,180,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                     
               
        named = "Croped_redFilter_Rotate270_Flip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
        named = "Croped_redFilter_Rotate270_noFlip" + name + order
        result = filter.redFilter(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
            #    
            #2
        named = "Croped_blueFilter_noRotate_noFlip" + name + order
        result = filter.blueFilter(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_blueFilter_noRotate_Flip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_blueFilter_Rotate90_Flip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_blueFilter_Rotate90_noFlip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_blueFilter_Rotate45_Flip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_blueFilter_Rotate45_noFlip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_blueFilter_Rotate180_Flip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,180,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_blueFilter_Rotate180_noFlip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,180,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                     
               
        named = "Croped_blueFilter_Rotate270_Flip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
        named = "Croped_blueFilter_Rotate270_noFlip" + name + order
        result = filter.blueFilter(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
             
            #2
            # 3
        named = "Croped_greenFilter_noRotate_noFlip" + name + order
        result = filter.greenFilter(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_greenFilter_noRotate_Flip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_greenFilter_Rotate90_Flip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_greenFilter_Rotate90_noFlip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_greenFilter_Rotate45_Flip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_greenFilter_Rotate45_noFlip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_greenFilter_Rotate180_Flip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,180,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_greenFilter_Rotate180_noFlip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,180,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                     
               
        named = "Croped_greenFilter_Rotate270_Flip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
        named = "Croped_greenFilter_Rotate270_noFlip" + name + order
        result = filter.greenFilter(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
            
            # 3   
            # 4
        named = "Croped_purpleFilter_noRotate_noFlip" + name + order
        result = filter.purpleFilter(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_purpleFilter_noRotate_Flip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_purpleFilter_Rotate90_Flip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_purpleFilter_Rotate90_noFlip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_purpleFilter_Rotate45_Flip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_purpleFilter_Rotate45_noFlip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_purpleFilter_Rotate180_Flip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,180,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_purpleFilter_Rotate180_noFlip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,180,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                     
               
        named = "Croped_purpleFilter_Rotate270_Flip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
        named = "Croped_purpleFilter_Rotate270_noFlip" + name + order
        result = filter.purpleFilter(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
            
            # 4
            
            # 5
        named = "Croped_yellowFilter_noRotate_noFlip" + name + order
        result = filter.yellowFilter(img)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_yellowFilter_noRotate_Flip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_yellowFilter_Rotate90_Flip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,90,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_yellowFilter_Rotate90_noFlip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,90,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_yellowFilter_Rotate45_Flip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,45,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_yellowFilter_Rotate45_noFlip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,45,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_yellowFilter_Rotate180_Flip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,180,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])          
        
        named = "Croped_yellowFilter_Rotate180_noFlip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,180,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                     
               
        named = "Croped_yellowFilter_Rotate270_Flip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,270,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
               
        named = "Croped_yellowFilter_Rotate270_noFlip" + name + order
        result = filter.yellowFilter(img)
        result = filter.originImage(result,270,False)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])                  
            
            # 5
               
        
        named = "Croped_ImgEdit_4x15_noRotate_noFlip" + name + order
        result = filter.ImageEditor(img, 1.5, 1.5, 1.5, 1.5, 0)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_ImgEdit_4x1_noRotate_noFlip" + name + order
        result = filter.ImageEditor(img, 1, 1, 1, 1, 0)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_ImgEdit_4x05_noRotate_noFlip" + name + order
        result = filter.ImageEditor(img, 0.5, 0.5, 0.5, 0.5, 0)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        
                   
        
        named = "Croped_ImgEdit_4x15_noRotate_Flip" + name + order
        result = filter.ImageEditor(img, 1.5, 1.5, 1.5, 1.5, 0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        
        named = "Croped_ImgEdit_4x1_noRotate_Flip" + name + order
        result = filter.ImageEditor(img, 1, 1, 1, 1, 0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        
        named = "Croped_ImgEdit_4x05_noRotate_Flip" + name + order
        result = filter.ImageEditor(img, 0.5, 0.5, 0.5, 0.5, 0,True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        named = "Croped_ImgEdit_4x15_Rotate90_Flip" + name + order
        result = filter.ImageEditor(img, 1.5, 1.5, 1.5, 1.5, 90, True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])
        named = "Croped_ImgEdit_4x1_Rotate90_Flip" + name + order
        result = filter.ImageEditor(img, 1, 1, 1, 1, 90, True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        named = "Croped_ImgEdit_4x05_Rotate90_Flip" + name + order
        result = filter.ImageEditor(img, 0.5, 0.5, 0.5, 0.5, 90, True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        named = "Croped_ImgEdit_4x15_Rotate45_Flip" + name + order
        result = filter.ImageEditor(img, 1.5, 1.5, 1.5, 1.5, 45, True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])  
        named = "Croped_ImgEdit_4x1_Rotate45_Flip" + name + order
        result = filter.ImageEditor(img, 1, 1, 1, 1, 45, True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        
        named = "Croped_ImgEdit_4x05_Rotate45_Flip" + name + order
        result = filter.ImageEditor(img, 0.5, 0.5, 0.5, 0.5, 45, True)
        cv2.imwrite(os.path.join(outputPath , named +'.jpg'), result,[cv2.IMWRITE_JPEG_QUALITY,80])        


def ImgGen(img, outputPath, count, outPutWidth=1280, outputHeight=720, reSize = False):
    if reSize is False:
        name = "noReSize"
        saveImg(img,name,outputPath,count)
    else:
        result = np.zeros((outputHeight, outPutWidth, 3), dtype="uint8")
        imgHeight, imgWidth = img.shape[:2]
        print("Image Size",imgHeight,imgWidth, " ",imgHeight*imgWidth," Mpx")
        rowCount = 1
        columnCount = 1
        if imgWidth > outPutWidth or imgHeight > outputHeight:
            columnCount = round(imgWidth/outPutWidth,0)
            rowCount = round(imgHeight/outputHeight,0)
            rowCount = int(rowCount)
            columnCount = int(columnCount)
        else:
            outputHeight = imgHeight
            outPutWidth = imgWidth
        ha = 0
        wa = 0
        limitXaxist = imgWidth - outPutWidth
        limitYaxist = imgHeight - outputHeight
        print("detect rows",str(rowCount),"detected columns",str(columnCount))
        print("Limit X",limitXaxist,"Limit Y",limitYaxist)
        countImg = 0
        for x in range(rowCount):
            for y in range(columnCount):                       
                wa = wa + (outPutWidth*y)
                if ha > limitYaxist:
                    ha = limitYaxist
                if wa > limitXaxist:
                    wa = limitXaxist             
                print("Start coordinates:",x,y,wa,ha)
                cropped_image = img[ha:ha+outputHeight, wa:wa+outPutWidth]
                h, w = cropped_image.shape[:2]
                print("Image is moved to coordinates"+str(ha),str(wa),str(h),str(w))
                result[0:h, 0:w] = cropped_image
                if y == (columnCount - 1) and x != (rowCount - 1):
                    wa = 0
                    ha = ha + outputHeight
                name = str(countImg)
                saveImg(result,name,outputPath,count)
                countImg = countImg + 1
