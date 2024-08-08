import torch
import torch.nn as nn
import numpy as np

def gaussian_kernel(size, std):
    ax = np.arange(-size // 2, size // 2)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * std**2))
    return kernel / kernel.sum()

class Net(nn.Module):
    def __init__(self, threshold, use_cuda=False):
        super(Net, self).__init__()
        self.threshold = threshold
        self.use_cuda = use_cuda

        # 生成高斯滤波器
        filter_size = 7
        std = 25
        generated_filters = gaussian_kernel(filter_size, std)
        generated_filters = torch.from_numpy(generated_filters).float()
        self.gaussian_filter = nn.Conv2d(1, 1, filter_size,
                                         padding=(filter_size//2), stride=1, bias=None)
        self.gaussian_filter.weight.data = generated_filters.unsqueeze_(0).unsqueeze_(0)
        # if self.use_cuda:
        #     self.gaussian_filter.weight.data = self.gaussian_filter.weight.data.to('cuda')

        # Sobel 滤波器
        sobel_filter_horizontal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
        sobel_filter_vertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        self.sobel_filter_horizontal = nn.Conv2d(1, 1, 3,
                                                 padding=1, stride=1, bias=None)
        self.sobel_filter_horizontal.weight.data = torch.from_numpy(sobel_filter_horizontal).float().unsqueeze_(0).unsqueeze_(0)
        # if self.use_cuda:
        #     self.sobel_filter_horizontal.weight.data = self.sobel_filter_horizontal.weight.data.to('cuda')

        self.sobel_filter_vertical = nn.Conv2d(1, 1, 3,
                                               padding=1, stride=1, bias=None)
        self.sobel_filter_vertical.weight.data = torch.from_numpy(sobel_filter_vertical).float().unsqueeze_(0).unsqueeze_(0)
        if self.use_cuda:
            self.sobel_filter_vertical.weight.data = self.sobel_filter_vertical.weight.data.to('cuda')

    def forward(self, img):
        if self.use_cuda and img.device.type != 'cuda':
            img = img.to('cuda')

        # 高斯模糊
        blurred_img = self.gaussian_filter(img)

        # Sobel 算子计算梯度
        grad_x = self.sobel_filter_horizontal(blurred_img)
        grad_y = self.sobel_filter_vertical(blurred_img)

        # 梯度幅度
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_orientation = torch.atan2(grad_y, grad_x)

        # 非极大值抑制
        max_kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=grad_mag.dtype, device=grad_mag.device)
        suppressed_x = torch.nn.functional.conv2d(grad_mag, max_kernel.unsqueeze(0).unsqueeze(0), padding=1)
        suppressed_y = torch.nn.functional.conv2d(grad_mag, max_kernel.transpose(0, 1).unsqueeze(0).unsqueeze(0), padding=1)
        thin_edges = torch.where((grad_mag >= suppressed_x) & (grad_mag >= suppressed_y), grad_mag, torch.tensor(0.0).to(grad_mag.device))

        # 双阈值检测
        strong_edges = thin_edges > (self.threshold * 3)
        weak_edges = (thin_edges > self.threshold) & (thin_edges <= (self.threshold * 3))

        # 进行Hysteresis Check
        strong_edges = strong_edges.type(torch.bool)
        weak_edges = weak_edges.type(torch.bool)
        edges = strong_edges | (weak_edges & self.hysteresis_check(weak_edges, strong_edges))

        return blurred_img, grad_mag, grad_orientation, thin_edges, edges

    def hysteresis_check(self, weak_edges, strong_edges):
        # 定义3x3的卷积核，用于Hysteresis检查
        kernel = torch.tensor([[1, 1, 1],
                               [1, -8, 1],
                               [1, 1, 1]], dtype=torch.float32)

        if self.use_cuda:
            kernel = kernel.to('cuda')

        kernel = kernel.unsqueeze(0).unsqueeze(0)

        # 执行卷积操作
        strong_convolved = torch.nn.functional.conv2d(strong_edges.float(), kernel, padding=1)

        # 将卷积结果转换为布尔型张量，表示是否有强边缘点
        strong_neighbors = (strong_convolved > 0).type(torch.bool)

        # 逻辑与操作，选择那些周围有强边缘的弱边缘点
        edges = weak_edges & strong_neighbors

        return edges

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    net = Net(threshold=10.0, use_cuda=use_cuda)
    net.eval()  # Set the network in evaluation mode

    # 假设有一个单通道的随机图像张量作为输入
    img = torch.randn(1, 1, 64, 64, dtype=torch.float32)
    if use_cuda:
        img = img.to('cuda')

    with torch.no_grad():  # No need to track gradients for inference
        outputs = net(img)  # 存储输出的元组
        blurred_img, grad_mag, grad_orientation, thin_edges, edges = outputs  # 解包元组

        # 打印每个输出的形状
        print("Blurred Image Shape:", blurred_img.shape)
        print("Gradient Magnitude Shape:", grad_mag.shape)
        print("Gradient Orientation Shape:", grad_orientation.shape)
        print("Thin Edges Shape:", thin_edges.shape)
        print("Edges Shape:", edges.shape)
