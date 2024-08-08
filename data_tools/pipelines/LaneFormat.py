from collections.abc import Sequence
import numpy as np
from ..DataContainer import DataContainer as DC
from .build import PIPELINES
import torch
import mmengine

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@PIPELINES.register_module()
class LaneFormat(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and other lane data. These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if len(img.shape) > 3:
                # [H, W, 3, N] -> [3, H, W, N]
                img = np.ascontiguousarray(img.transpose(2, 0, 1, 3))
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_3dlanes' in results:
            results['gt_3dlanes'] = DC(to_tensor(results['gt_3dlanes'].astype(np.float32)))
        if 'gt_2dlanes' in results:
            results['gt_2dlanes'] = DC(to_tensor(results['gt_2dlanes'].astype(np.float32)))
        if 'gt_camera_extrinsic' in results:
            results['gt_camera_extrinsic'] = DC(to_tensor(results['gt_camera_extrinsic'][None, ...].astype(np.float32)), stack=True)
        if 'gt_camera_intrinsic' in results:
            results['gt_camera_intrinsic'] = DC(to_tensor(results['gt_camera_intrinsic'][None, ...].astype(np.float32)), stack=True)
        if 'gt_project_matrix' in results:
            results['gt_project_matrix'] = DC(to_tensor(results['gt_project_matrix'][None, ...].astype(np.float32)), stack=True)
        if 'gt_homography_matrix' in results:
            results['gt_homography_matrix'] = DC(to_tensor(results['gt_homography_matrix'][None, ...].astype(np.float32)), stack=True)
        if 'gt_camera_pitch' in results:
            results['gt_camera_pitch'] = DC(to_tensor([results['gt_camera_pitch']]))
        if 'gt_camera_height' in results:
            results['gt_camera_height'] = DC(to_tensor([results['gt_camera_height']]))
        if 'prev_poses' in results:
            results['prev_poses'] = DC(to_tensor(np.stack(results['prev_poses'], axis=0).astype(np.float32)), stack=True)  # [Np, 3, 4]
        if 'mask' in results:
            results['mask'] = DC(to_tensor(results['mask'][None, ...].astype(np.float32)), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
