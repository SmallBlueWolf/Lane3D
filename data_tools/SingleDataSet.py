import os
from PIL import Image

from torch.utils.data import Dataset
from .build import DATASETS
from .pipelines.Compose import Compose

@DATASETS.register_module()
class SingleDataset(Dataset):
    def __init__(self, 
                 pipeline,
                 data_root="",
                 img_dir="", 
                 img_suffix='.jpg',
                 test_mode=True,
                 dataset_config=None,
                 is_resample=True):
        # 保留原始数据集定义中的预处理部分
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.img_suffix = img_suffix
        self.test_mode = test_mode  # 设置为True，表示仅用于推理
        self.is_resample = is_resample
        self.dataset_config = dataset_config

        # 预处理参数
        inp_h, inp_w = dataset_config['input_size']
        self.h_org = 1080
        self.w_org = 1920
        self.h_net = inp_h
        self.w_net = inp_w
        self.resize_h = inp_h
        self.resize_w = inp_w
        # ... 其他相关参数设置

        # 初始化图像信息列表
        self.img_infos = self._load_image_infos()

    def _load_image_infos(self):
        """加载图片信息，不加载注释或标签"""
        img_infos = []
        all_ids = os.listdir(self.img_dir)
        for id in all_ids:
            filename = os.path.join(self.img_dir, id)
            img_infos.append({'filename': filename})
        return img_infos

    def __getitem__(self, idx, transform=False):
        """获取推理数据"""
        results = self.img_infos[idx].copy()
        results['img_info'] = {}
        results['img_info']['filename'] = results['filename']
        results['ori_shape'] = (self.h_org, self.w_org)  # 原始图像尺寸
        results['flip'] = False
        results['flip_direction'] = None
        results['img_metas'] = {'ori_shape':results['ori_shape']}
        # 由于是推理模式，不加载注释或标签
        return self.pipeline(results)

    def __len__(self):
        return len(self.img_infos)