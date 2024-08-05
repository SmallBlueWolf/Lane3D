from .build import PIPELINES
from typing import Callable, Type, Union
import numpy as np
import torch
import functools
from collections.abc import Sequence
import mmcv

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
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def assert_tensor_type(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper

class DC:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data: Union[torch.Tensor, np.ndarray],
                 stack: bool = False,
                 padding_value: int = 0,
                 cpu_only: bool = False,
                 pad_dims: int = 2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> int:
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs) -> torch.Size:
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self) -> int:
        return self.data.dim()

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
            results['img'] = to_tensor(img)
        if 'gt_3dlanes' in results:
            results['gt_3dlanes'] = to_tensor(results['gt_3dlanes'].astype(np.float32))
        if 'gt_2dlanes' in results:
            results['gt_2dlanes'] = to_tensor(results['gt_2dlanes'].astype(np.float32))
        if 'gt_camera_extrinsic' in results:
            results['gt_camera_extrinsic'] = to_tensor(results['gt_camera_extrinsic'][None, ...].astype(np.float32))
        if 'gt_camera_intrinsic' in results:
            results['gt_camera_intrinsic'] = to_tensor(results['gt_camera_intrinsic'][None, ...].astype(np.float32))
        if 'gt_project_matrix' in results:
            results['gt_project_matrix'] = to_tensor(results['gt_project_matrix'][None, ...].astype(np.float32))
        if 'gt_homography_matrix' in results:
            results['gt_homography_matrix'] = to_tensor(results['gt_homography_matrix'][None, ...].astype(np.float32))
        if 'gt_camera_pitch' in results:
            results['gt_camera_pitch'] = to_tensor([results['gt_camera_pitch']])
        if 'gt_camera_height' in results:
            results['gt_camera_height'] = to_tensor([results['gt_camera_height']])
        if 'prev_poses' in results:
            results['prev_poses'] = to_tensor(np.stack(results['prev_poses'], axis=0).astype(np.float32))  # [Np, 3, 4]
        if 'mask' in results:
            results['mask'] = to_tensor(results['mask'][None, ...].astype(np.float32))
        return results

    def __repr__(self):
        return self.__class__.__name__