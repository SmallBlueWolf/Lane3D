import cv2
import numpy as np

THRESHOLD_DELTA = 1

import copy
import numpy as np


def rgb_to_ycbcr(im_rgb):
    im_yuv = copy.deepcopy(im_rgb)

    R = im_rgb[:, :, 0]
    G = im_rgb[:, :, 1]
    B = im_rgb[:, :, 2]

    im_yuv[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    im_yuv[:, :, 1] = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    im_yuv[:, :, 2] = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    return np.uint8(im_yuv)

def isImgYellowLine(img):
    size = img.shape
    height = size[0]
    targetY = []
    
    for i in range(-THRESHOLD_DELTA,THRESHOLD_DELTA+1):
        targetY.append(int(height*3/4) + i)
    yuv_img = rgb_to_ycbcr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).astype(np.int64)
    for y in targetY:
        for x in range(height):
            U = yuv_img[x,y,1]
            V = yuv_img[x,y,2]
            if U-V<-15:
                return True
    return False