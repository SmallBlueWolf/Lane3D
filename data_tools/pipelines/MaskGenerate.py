from .build import PIPELINES
import numpy as np

@PIPELINES.register_module()
class MaskGenerate(object):
    def __init__(self, input_size):
        self.input_size = input_size 

    def __call__(self, results):
        mask  = np.ones((self.input_size[0], self.input_size[1]), dtype=bool)
        mask = np.logical_not(mask)
        results['mask'] = mask
        return results