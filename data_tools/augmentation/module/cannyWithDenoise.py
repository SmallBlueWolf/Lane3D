from .canny.net_canny import Net
import torch
import torch.nn as nn
import numpy as np

def canny(raw_img, use_cuda=False, threshold=0.05):
    img_gray = np.dot(raw_img[...,:3], [0.2989, 0.587, 0.114])
    img = torch.from_numpy(img_gray).unsqueeze(0).float()
    
    net = Net(threshold=threshold, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()
    
    data = img.unsqueeze(0)
    if use_cuda:
        data = data.cuda()
        
    with torch.no_grad():
        outputs = net(data)
        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded = outputs
    
    grad_mag_np = grad_mag.data.cpu().numpy()[0, 0].astype(np.float32)  # 确保是浮点数
    grad_mag_8bit = (grad_mag_np - grad_mag_np.min()) * (255.0 / (grad_mag_np.max() - grad_mag_np.min()))  # 缩放到0-255
    grad_mag_8bit = grad_mag_8bit.astype(np.uint8)  # 转换为8位整数
    
    thresholded_np = thresholded.data.cpu().numpy()[0, 0]
    # print("Final Edges Nonzero:", np.count_nonzero(final_edges))
    thresholded_8bit = (thresholded_np * 255).astype(np.uint8)
    
    return thresholded_8bit