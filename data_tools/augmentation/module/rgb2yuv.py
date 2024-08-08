import copy
import numpy as np


def rgb_to_ycbcr(im_rgb):
    im_yuv = copy.deepcopy(im_rgb)

    R = im_rgb[:, :, 0]
    G = im_rgb[:, :, 1]
    B = im_rgb[:, :, 2]

    im_yuv[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    im_yuv[:, :, 1] = -0.1687 * R - 0.3313 * G + 0.5 * B
    im_yuv[:, :, 2] = 0.5 * R - 0.4187 * G - 0.0813 * B

    return np.uint8(im_yuv)