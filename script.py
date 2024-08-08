import torch
import cv2
import numpy as np
from tqdm import tqdm
import glob

# 获取所有图像路径
l = glob.glob("data/images(depth)/training/seg*/*.png")
img_list = l + glob.glob("data/images(depth)/validation/seg*/*.png")

# 初始化累加器
mean = torch.zeros(1, dtype=torch.float32, device='cuda')
stdev = torch.zeros(1, dtype=torch.float32, device='cuda')

# 遍历所有图像
for imgs_path in tqdm(img_list):
    img = cv2.imread(imgs_path, cv2.IMREAD_GRAYSCALE)  # 以灰度图方式读取图像
    img = torch.tensor(img, dtype=torch.float32, device='cuda')  # 转为torch tensor，并移动到GPU

    # 计算像素值总和
    mean += img.mean()
    stdev += img.std()

# 计算均值和标准差
num_imgs = len(img_list)
mean /= num_imgs
stdev /= num_imgs

# 从GPU移动到CPU并转换为numpy
mean = mean.cpu().numpy()
stdev = stdev.cpu().numpy()

print(f"Mean: {mean}")
print(f"Stddev: {stdev}")
