
from mmengine import Config
from models.componets.build import LANENET2S
from mmengine.runner import load_checkpoint
import mmcv
import os
import torch
import tqdm
import json
import torch.nn.functional as F
from models.Anchor3DLane import Anchor3DLane
from models.componets.backbones.resnet import ResNetV1c
from models.componets.losses.lane_loss import LaneLoss
from models.componets.assigner.topk_assigner import TopkAssigner

if __name__ == '__main__':
    cfg = Config.fromfile('config.py')
    model = LANENET2S.build(cfg.model)
    print(model)