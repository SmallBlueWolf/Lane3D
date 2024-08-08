from mmengine import Config, build_from_cfg
from mmengine.runner import load_checkpoint
from models.build import MODELS
from data_tools.build import DATASETS
from data_tools.builder import build_dataloader, get_device
from data_tools.OpenlaneDataset import OpenlaneDataset
from mmengine.model import revert_sync_batchnorm
from models.collate import collate
from inference_module import build_dp, inference_openlane, evaluation

if __name__ == '__main__':
    cfg = Config.fromfile('config.py')
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    model = build_dp(model, get_device(), device_ids=[0])
    
    checkpoint = load_checkpoint(model, "/media/bluewolf/Data/bluewolf/projs/Anchor3D/output/openlane/anchor3dlane/iter_60000.pth", map_location='cpu')
    
    dataset = build_from_cfg(cfg.data.test, DATASETS)
    test_loader_cfg = {
        'num_gpus': 1,
        'dist': False,
        'workers_per_gpu': 4,
        'samples_per_gpu': 16,    # for debug
        'shuffle': False  # Not shuffle by default
    }
    data_loader = build_dataloader(dataset=dataset, custom_collate=collate, **test_loader_cfg)
    inference_openlane(model, data_loader, "./output(3)")
    evaluation(data_loader, "./output(3)", 0.5, r"./data/ew.json")