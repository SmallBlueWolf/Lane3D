import argparse
import mmcv
import torch
from mmseg.apis import inference, init_detector, show_result

def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--config', default='config.py', help='config file path')
    parser.add_argument('--weight', default='model.pth', help='weight file path')
    parser.add_argument('--source', help='input image path')
    parser.add_argument('--device', type = str, default='cuda:0', help='device')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = init_detector(args.config, args.weight, device=args.device)