import cv2
from .module.rgb2yuv import rgb_to_ycbcr
from .module.cannyWithDenoise import canny
from .module.checkY import isImgYellowLine
import numpy as np

def add_images(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    added_image = cv2.add(image1, image2)
    added_image = np.clip(added_image, 0, 255).astype(np.uint8)
    return added_image

def augumentation(img, COLOR_TRANS_FLAG=False, use_cuda=False):
    if COLOR_TRANS_FLAG:
        newimg = rgb_to_ycbcr(img)
    newimg = canny(img/255.0, use_cuda=use_cuda)
    newimg = cv2.cvtColor(newimg, cv2.IMREAD_COLOR)
    img = add_images(img, newimg)
    COLOR_TRANS_FLAG = isImgYellowLine(img)
    return img

if __name__ == '__main__':
    img = cv2.imread('./SDG.png')
    img = augumentation(img)
    cv2.imwrite('test_result2.jpg', img)