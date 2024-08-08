import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def add_images(image1, image2):
    """
    将两张RGB图像的像素值相加。
    :param image1: 第一张RGB图像。
    :param image2: 第二张RGB图像。
    :return: 相加后的图像。
    """
    # 确保两个图像尺寸相同
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")

    # 将两个图像的像素值相加
    added_image = cv2.add(image1, image2)

    # 确保像素值不超过255
    added_image = np.clip(added_image, 0, 255).astype(np.uint8)

    return added_image


if __name__ == '__main__':
    # 读取两张原始图像
    image1 = cv2.imread('adverse.jpeg')
    image2 = cv2.imread('final.png')
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # 确保两张图像都成功读取
    if image1 is None or image2 is None:
        print("Error: Images could not be read.")
    else:
        # 图像相加
        result_image = add_images(image1_rgb, image2_rgb)

        # 显示结果
        plt.subplot(2, 2, 1)
        plt.title('origin image')
        plt.imshow(image1_rgb)

        plt.subplot(2, 2, 2)
        plt.title('canny edge')
        plt.imshow(image2_rgb)

        plt.subplot(2, 2, 3)
        plt.title('final image')
        plt.imshow(result_image)

        plt.show()