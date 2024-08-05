from mmengine import Registry, build_from_cfg
import torch

def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s)."""
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return torch.nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

MODELS = Registry('models', build_func=build_model_from_cfg)

ASSIGNER = Registry('assigner')

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
LANENET2S = MODELS

def build_assigner(cfg):
    """Build anchor-gt matching function"""
    return ASSIGNER.build(cfg)

def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)

def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)

# def build_head(cfg):
#     """Build head."""
#     return HEADS.build(cfg)

def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_lanedetector(cfg):
    return LANENET2S.build(cfg)