import cv2
import numpy as np
from PIL import Image, ImageEnhance

originalValues = np.array([0, 50, 100, 150, 200, 255])


def Image2Grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def changesSharpness(img, Sharpness=1):
    img = Image.fromarray(img)
    filter = ImageEnhance.Sharpness(img)
    img = filter.enhance(Sharpness)
    img = np.array(img)
    return img


def changesContrast(img, Contrast=1):
    img = Image.fromarray(img)
    filter = ImageEnhance.Contrast(img)
    img = filter.enhance(Contrast)
    img = np.array(img)
    return img


def changesBrightness(img, Brightness=1):
    img = Image.fromarray(img)
    filter = ImageEnhance.Brightness(img)
    img = filter.enhance(Brightness)
    img = np.array(img)
    return img


def changesSaturation(img, Saturation=1):
    img = Image.fromarray(img)
    filter = ImageEnhance.Color(img)
    img = filter.enhance(Saturation)
    img = np.array(img)
    return img


def originImage(img, angle=0, areFlip=False):
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def vio01Filter(img, angle=0, areFlip=False):
    Values1 = np.array([0, 50, 100, 100, 150, 255])
    allValues = np.arange(0, 256)
    greenLookupTable = np.interp(allValues, originalValues, Values1)
    allValues = np.arange(0, 256)
    greenLookupTable = np.interp(allValues, originalValues, Values1)
    B, G, R = cv2.split(img)

    G = cv2.LUT(B, greenLookupTable)
    G = np.uint8(G)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def yellowFilter(img, angle=0, areFlip=False):

    redValues = np.array([0, 80, 150, 190, 250, 255])
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)

    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(img)
    R = cv2.LUT(R, redLookupTable)
    R = np.uint8(R)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def greenFilter(img, angle=0, areFlip=False):

    redValues = np.array([0, 0, 0, 0, 0, 0])
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)

    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(img)
    R = cv2.LUT(R, redLookupTable)
    R = np.uint8(R)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def redFilter(img, angle=0, areFlip=False):

    redValues = np.array([0, 0, 0, 0, 0, 0])
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)

    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(img)
    G = cv2.LUT(G, redLookupTable)
    G = np.uint8(G)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def purpleFilter(img, angle=0, areFlip=False):
    redValues = np.array([0, 0, 10, 10, 20, 25])
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)

    B, G, R = cv2.split(img)
    G = cv2.LUT(G, redLookupTable)
    G = np.uint8(G)

    img = cv2.merge([B, G, B])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def blueFilter(img, angle=0, areFlip=False):
    Values1 = np.array([0, 0, 0, 0, 0, 0])
    Values2 = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)
    greenLookupTable = np.interp(allValues, originalValues, Values1)
    redLookupTable = np.interp(allValues, originalValues, Values2)
    allValues = np.arange(0, 256)
    greenLookupTable = np.interp(allValues, originalValues, Values1)
    redLookupTable = np.interp(allValues, originalValues, Values2)

    B, G, R = cv2.split(img)
    G = cv2.LUT(G, greenLookupTable)
    G = np.uint8(G)

    R = cv2.LUT(R, redLookupTable)
    R = np.uint8(R)

    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def RedAndBlueFilter(img, angle=0, areFlip=False):
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(img)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def GreenAndBlueFilter(img, angle=0, areFlip=False):
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    B, G, R = cv2.split(img)

    R = cv2.LUT(R, blueLookupTable)
    R = np.uint8(R)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def BlueAndRedFilter(img, angle=0, areFlip=False):
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(img)

    G = cv2.LUT(G, blueLookupTable)
    G = np.uint8(G)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))
    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def testFilter(img, angle=0, areFlip=False):
    redValues = np.array([0, 80, 150, 190, 250, 255])
    blueValues = np.array([0, 0, 0, 0, 0, 0])
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)
    allValues = np.arange(0, 256)
    redLookupTable = np.interp(allValues, originalValues, redValues)
    blueLookupTable = np.interp(allValues, originalValues, blueValues)

    B, G, R = cv2.split(img)
    R = cv2.LUT(R, redLookupTable)
    R = np.uint8(R)

    B = cv2.LUT(B, blueLookupTable)
    B = np.uint8(B)
    img = cv2.merge([B, G, R])
    height, width = img.shape[:2]
    center = (width/2, height/2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(
        src=img, M=rotate_matrix, dsize=(width, height))

    if areFlip is True:
        rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image


def ImageEditor(img, Brightness=1, Saturation=1, Sharpness=1, Contrast=1, Angle=1, Flip=False):
    img = changesBrightness(img, Brightness)
    img = changesSaturation(img, Saturation)
    img = changesSharpness(img, Sharpness)
    img = changesContrast(img, Contrast)
    img = originImage(img, Angle, Flip)
    return img

# img = np.copy(image)
# cv2.namedWindow("img2", cv2.WINDOW_NORMAL)


# cv2.imshow("img2", ImageEditor(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
