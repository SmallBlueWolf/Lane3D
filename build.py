from mmengine import Registry, build_from_cfg
import torch

def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s)."""
    if isinstance(cfg, list):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return torch.nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

MODELS = Registry('model', build_func=build_model_from_cfg)

