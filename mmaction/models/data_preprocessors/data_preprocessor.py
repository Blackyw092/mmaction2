# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmaction.registry import MODELS
from mmaction.utils import SampleList


@MODELS.register_module()
class ActionDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for action recognition tasks.

    Args:
        mean (Sequence[float or int], 可选): 图像或堆叠光流通道的像素平均值。默认为 None。
        std (Sequence[float or int], 可选): 图像或堆叠光流通道的像素标准差。默认为 None。
        to_rgb (bool): 是否将图像从BGR转换为RGB。默认为 False。
        to_float32 (bool): 是否将数据转换为 float32。默认为 True。
        blending (dict, 可选): 批处理混合的配置。默认为 None。
        format_shape (str): 输入数据的格式形状。默认为'NCHW'
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 blending: Optional[dict] = None,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

        if blending is not None:
            self.blending = MODELS.build(blending)
        else:
            self.blending = None

    def forward(self,
                data: Union[dict, Tuple[dict]],
                training: bool = False) -> Union[dict, Tuple[dict]]:
        """基于 ``BaseDataPreprocessor`` 执行标准化、填充、BGR转RGB及批量增强。
           参数:
               data (dict or Tuple[dict]): 从数据加载器采样的数据。
               training (bool): 是否启用训练时间增强。

           返回:
               dict or Tuple[dict]: 与模型输入格式相同的数据显示。
         """
        data = self.cast_data(data)
        if isinstance(data, dict):
            return self.forward_onesample(data, training=training)
        elif isinstance(data, (tuple, list)):
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, training: bool = False) -> dict:
        """ 对单个数据样本执行标准化、填充、BGR转RGB及批量增强。
        参数:
            data (dict): 从数据加载器采样的数据。
            training (bool): 是否启用训练时间增强。

        返回:
            dict: 与模型输入格式相同的数据显示。
        """
        inputs, data_samples = data['inputs'], data['data_samples']
        inputs, data_samples = self.preprocess(inputs, data_samples, training)
        data['inputs'] = inputs
        data['data_samples'] = data_samples
        return data

    def preprocess(self,
                   inputs: List[torch.Tensor],
                   data_samples: SampleList,
                   training: bool = False) -> Tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == 'MIX2d3d':
            if batch_inputs.ndim == 4:
                format_shape, view_shape = 'NCHW', (-1, 1, 1)
            else:
                format_shape, view_shape = 'NCTHW', None
        else:
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == 'NCHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == 'NCTHW':
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

        # -- Normalization ---
        if self._enable_normalize:
            if view_shape is None:
                batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                mean = self.mean.view(view_shape)
                std = self.std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            batch_inputs = batch_inputs.to(torch.float32)

        # ----- Blending -----
        if training and self.blending is not None:
            batch_inputs, data_samples = self.blending(batch_inputs,
                                                       data_samples)

        return batch_inputs, data_samples
