# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class Video_Sk_Dataset(BaseActionDataset):
    """ 用于动作识别的原始帧数据集。

    该数据集加载原始帧并应用指定的变换，返回一个包含帧张量和其他信息的字典。

    ann_file 是一个文本文件，每行指示一个视频的帧目录、视频的总帧数和视频的标签，
    它们通过空格分隔。

    Args:
        ann_file (str): 注释文件的路径。
        pipeline (List[Union[dict, ConfigDict, Callable]]): 数据变换的序列。
        data_prefix (dict or ConfigDict): 视频帧所在的目录路径。默认为 ``dict(img='')``.
        filename_tmpl (str): 每个文件名的模板。默认为 ``img_{:05}.jpg``。
        with_offset (bool): 决定ann_file中是否包含偏移信息。默认为 False。
        multi_class (bool): 决定是否为多类别识别数据集。默认为 False。
        num_classes (int, optional): 数据集中类的数量。默认为 None。
        start_index (int): 考虑到不同的文件名格式，为帧指定一个起始索引。然而，当以帧为输入时，
            应该设置为 1，因为原始帧从 1 开始计数。默认为 1。
        modality (str): 数据的模态。支持 ``RGB``、``Flow``。默认为 ``RGB``。
        test_mode (bool): 构建测试或验证数据集时存储 True。默认为 False。
    """

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(img=''),
                 filename_tmpl: str = '{:06}.jpg',
                 with_offset: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 **kwargs) -> None:
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            video_info = {}

            idx = 0
            # idx for frame_dir
            rgb_frame_dir = line_split[idx]
            video_info['rgb_frame_dir'] = rgb_frame_dir

            idx += 1
            video_info['rgb_total_frames'] = int(line_split[idx])

            idx += 1
            sk_frame_dir = line_split[idx]
            video_info['sk_frame_dir'] = sk_frame_dir

            idx += 1
            video_info['sk_total_frames'] = int(line_split[idx])

            # idx for label[s]
            idx += 1
            video_info['label'] = int(line_split[idx])

            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['filename_tmpl'] = self.filename_tmpl
        return data_info



