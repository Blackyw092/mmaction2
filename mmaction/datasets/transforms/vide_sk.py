import copy as cp
import random
import io
import os
import os.path as osp
import shutil
from typing import Dict, List, Optional, Union

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
import mmengine
from mmengine.fileio import FileClient

from mmaction.registry import TRANSFORMS
from torch.nn.modules.utils import _pair
from mmaction.utils import get_random_string, get_shm_dir, get_thread_id

def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'rgb_img_shape' not in results:
        results['rgb_img_shape'] = results['rgb_imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['rgb_img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['rgb_img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'

@TRANSFORMS.register_module()
class Video_Sk_SampleFrames(BaseTransform):
    """从视频中采样帧。

    必需的键:
        - total_frames: 视频的总帧数。
        - start_index: 开始采样的帧索引。

    添加的键:
        - frame_inds: 采样帧的索引。
        - frame_interval: 相邻采样帧的时间间隔。
        - num_clips: 采样的片段数。


    参数:
        clip_len (int): 每个采样输出片段的帧数。
        frame_interval (int): 相邻采样帧的临时间隔。默认为1。
        num_clips (int): 要采样的片段数。默认为1。
        temporal_jitter (bool): 是否应用临时抖动。默认为False。
        twice_sample (bool): 是否在测试时使用两次采样。如果设置为True，它将以固定偏移量的方式采样帧，这是TSM模型测试中常用的。默认为False。
        out_of_bound_opt (str): 处理超出边界的帧索引的方法。可用选项为'loop', 'repeat_last'。
        test_mode (bool): 构建测试或验证数据集时存储True。默认为False。
        keep_tail_frames (bool): 采样时是否保留尾部帧。默认为False。
        target_fps (optional, int): 将输入视频的任意帧率转换为统一的目标FPS，然后在采样帧之前。如果为，则不会调整帧率。默认为``None``。
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 num_segments = 8,
                 **kwargs) -> None:

        """
        初始化SampleFrames类。

        :param clip_len: 每个采样片段的帧数。
        :param frame_interval: 采样帧之间的临时间隔。
        :param num_clips: 采样片段的数量。
        :param temporal_jitter: 是否应用时间抖动。
        :param twice_sample: 是否在测试模式下进行二次采样。
        :param out_of_bound_opt: 超出边界时的处理选项。
        :param test_mode: 是否为测试/验证模式。
        :param keep_tail_frames: 采样时是否保留尾帧。
        :param target_fps: 目标帧率，用于将输入视频统一转换到此帧率。
        """

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps

        self.num_segments = num_segments
        self.loop = False
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """
          训练模式下获取片段偏移量。

          计算选定帧的平均间隔，并在[0, avg_interval]范围内随机偏移它们。
          如果总帧数小于片段数或原始采样片段长度，则返回所有零索引。

          :param num_frames: 视频的总帧数。
          :param ori_clip_len: 原始采样片段的长度。
          :return: 训练模式下采样的帧索引。
          """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """
         测试模式下获取片段偏移量。

         如果总帧数不够，返回所有零索引。

         :param num_frames: 视频的总帧数。
         :param ori_clip_len: 原始采样片段的长度。
         :return: 测试模式下采样的帧索引。
         """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """
          根据给定模式为视频选择片段偏移量。

          :param num_frames: 视频的总帧数。
          :return: 采样的帧索引。
          """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """ 计算不同策略下的片段长度。

          :param fps_scale_ratio: 调整fps的缩放比例。
          :return: 计算得到的片段长度。
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len
#region test
    # def _sample_indices(self, num_frames: int,):
    #     # 采样视频帧 （等比例间隔后，加上随机和偏移）
    #     if num_frames <= self.num_segments:  # 判断视频长度是否小于总输入帧数
    #         if self.loop:
    #             # 是否开启循环补帧
    #             return np.mod(np.arange(
    #                 self.num_segments) + randint(num_frames // 2),
    #                           num_frames) + self.index_bias
    #         offsets = np.concatenate((
    #             np.arange(record.num_frames),
    #             randint(record.num_frames,
    #                     size=self.num_segments - num_frames)))
    #         return np.sort(offsets) + self.index_bias
    #
    #     offsets = list()
    #     # 等比例间隔取帧
    #     ticks = [i * record.num_frames // num_segments
    #              for i in range(num_segments + 1)]
    #
    #     for i in range(num_segments):
    #         tick_len = ticks[i + 1] - ticks[i]  # 帧之间间隔长度
    #         tick = ticks[i]  # 标签
    #         if tick_len >= self.seg_length:  # 如果间隔长度大于截取长度
    #             tick += randint(tick_len - self.seg_length + 1)  # 随机偏移
    #         offsets.extend([j for j in range(tick, tick + self.seg_length)])
    #     return np.array(offsets) + self.index_bias  # 采取了广播加偏移
    #
    # def _get_val_indices(self, record,type):
    #     # 验证集采样帧数
    #     if type == 'sk':
    #         num_segments = self.num_segments
    #     elif type == 'rgb':
    #         num_segments = self.num_segments
    #     if num_segments == 1:  # 输入帧数为一时，直接取中间一帧并加上偏移
    #         return np.array([record.num_frames // 2], dtype=np.int) + self.index_bias
    #
    #     if record.num_frames <= self.total_length:  # 判断视频长度是否小于总输入帧数
    #         if self.loop:
    #             return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
    #         return np.array([i * record.num_frames // self.total_length
    #                          for i in range(self.total_length)], dtype=np.int) + self.index_bias
    #
    #     offset = (record.num_frames / num_segments - self.seg_length) / 2.0
    #
    #     # 返回indexes
    #     return np.array([i * record.num_frames / num_segments+ offset + j
    #                      for i in range(num_segments)
    #                      for j in range(self.seg_length)], dtype=np.int) + self.index_bias
#endregion
    def transform(self, results: dict) -> dict:
        """ 执行SampleFrames加载。

        :param results: 由管道中下一个变换修改并传递的结果字典。
        :return: 修改后的结果字典。
        """
        rgb_total_frames = results['rgb_total_frames']
        sk_total_frames = results['sk_total_frames']

        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps

        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)


        rgb_clip_offsets = self._sample_clips(rgb_total_frames, ori_clip_len)
        sk_clip_offsets = self._sample_clips(sk_total_frames, ori_clip_len)

        # rgb流index
        if self.target_fps:
            rgb_frame_inds = rgb_clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            rgb_frame_inds = rgb_clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            rgb_frame_inds = np.concatenate(rgb_frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(rgb_frame_inds))
            rgb_frame_inds += perframe_offsets

        rgb_frame_inds = rgb_frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            rgb_frame_inds = np.mod(rgb_frame_inds, rgb_total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = rgb_frame_inds < rgb_total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * rgb_frame_inds, axis=1)
            new_inds = (safe_inds * rgb_frame_inds + (unsafe_inds.T * last_ind).T)
            rgb_frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        # sk流index
        if self.target_fps:
            sk_frame_inds = sk_clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            sk_frame_inds = sk_clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            sk_frame_inds = np.concatenate(rgb_frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(sk_frame_inds))
            sk_frame_inds += perframe_offsets

        sk_frame_inds = sk_frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            sk_frame_inds = np.mod(sk_frame_inds, sk_total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = sk_frame_inds < sk_total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * sk_frame_inds, axis=1)
            new_inds = (safe_inds * sk_frame_inds + (unsafe_inds.T * last_ind).T)
            sk_frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        rgb_frame_inds = np.concatenate(rgb_frame_inds) + start_index
        sk_frame_inds = np.concatenate(sk_frame_inds) + start_index

        results['rgb_frame_inds'] = rgb_frame_inds.astype(np.int32)
        results['sk_frame_inds'] = sk_frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

@TRANSFORMS.register_module()
class Video_Sk_RawFrameDecode(BaseTransform):
    """加载并解码给定索引的帧。

    必需的键:
    - frame_dir: 帧目录路径。
    - filename_tmpl: 文件名模板。
    - frame_inds: 帧索引。
    - modality: 数据模态，例如'RGB'或'Flow'。
    - offset (可选): 帧索引的偏移量。

    添加的键:
    - img: 解码后的图像。
    - img_shape: 图像的形状（高度和宽度）。
    - original_shape: 原始图像的形状（高度和宽度）。

    Args:
        io_backend (str): io_backend (str): 帧存储的IO后端。
            Defaults to ``'disk'``.
        decoding_backend (str):  图像解码的后端。
            Defaults to ``'cv2'``.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        # 初始化类变量
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 设置解码时使用的后端
        mmcv.use_backend(self.decoding_backend)

        rgb_directory = results['rgb_frame_dir']
        sk_directory = results['sk_frame_dir']

        filename_tmpl = results['filename_tmpl']

        # 如果文件客户端未初始化，则进行初始化
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        rgb_imgs = list()
        sk_imgs = list()

        # 确保帧索引是一维的
        if results['rgb_frame_inds'].ndim != 1:
            results['rgb_frame_inds'] = np.squeeze(results['rgb_frame_inds'])
        if results['sk_frame_inds'].ndim != 1:
            results['sk_frame_inds'] = np.squeeze(results['sk_frame_inds'])


        # 获取可能存在的帧索引偏移量
        offset = results.get('offset', 0)

        cache = {}  # 用于缓存已加载的帧，避免重复加载
        for i, frame_idx in enumerate(results['rgb_frame_inds']):
            # 如果帧已缓存，则直接使用缓存的副本
            if frame_idx in cache:
                rgb_imgs.append(cp.deepcopy(rgb_imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i
            # 计算实际要加载的帧索引
            frame_idx += offset
            filepath = osp.join(rgb_directory, filename_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            # 直接以RGB顺序加载帧
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            rgb_imgs.append(cur_frame)


        cache = {}  # 用于缓存已加载的帧，避免重复加载
        for i, frame_idx in enumerate(results['sk_frame_inds']):
            # 如果帧已缓存，则直接使用缓存的副本
            if frame_idx in cache:
                sk_imgs.append(cp.deepcopy(sk_imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i

            # 计算实际要加载的帧索引
            frame_idx += offset
            # 根据模态加载和处理图像
            filepath = osp.join(sk_directory, filename_tmpl.format(frame_idx))
            img_bytes = self.file_client.get(filepath)
            # 直接以RGB顺序加载帧
            cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            sk_imgs.append(cur_frame)

        # 更新结果字典以包含加载的图像和其他相关信息
        results['rgb_imgs'] = rgb_imgs
        results['rgb_original_shape'] = rgb_imgs[0].shape[:2]
        results['rgb_img_shape'] = rgb_imgs[0].shape[:2]

        results['sk_imgs'] =sk_imgs
        results['sk_original_shape'] = sk_imgs[0].shape[:2]
        results['sk_img_shape'] = sk_imgs[0].shape[:2]

        return results

    def __repr__(self):
        # 生成类的字符串表示形式
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str

@TRANSFORMS.register_module()
class  Video_Sk_Resize(BaseTransform):
    """调整图像到特定尺寸的函数。

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        - `scale`：如果`keep_ratio`为真，则作为缩放因子或最大尺寸：
          - 若为浮点数，图像将按此比例缩放；
          - 若为包含两个整数的元组，图像将被缩放到能在该尺度内尽可能大的尺寸。
          - 否则，它表示输出图像的宽（w）和高（h）。
        - `keep_ratio`：若设为True，图像将在不失真的情况下进行缩放；否则，将图像调整到给定尺寸。默认值：True。
        - `interpolation`：使用的插值算法，可接受的值有"nearest"、"bilinear"、"bicubic"、"area"、"lanczos"。默认值："bilinear"。
        - `lazy`：确定是否应用延迟操作。默认值：False。
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        # 确定比例
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale

        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]
    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'rgb_scale_factor' not in results:
            results['rgb_scale_factor'] = np.array([1, 1], dtype=np.float32)
            results['sk_scale_factor'] = np.array([1, 1], dtype=np.float32)
        rgb_img_h, rgb_img_w = results['rgb_img_shape']
        sk_img_h, sk_img_w = results['sk_img_shape']

        if self.keep_ratio:
            rgb_new_w, rgb_new_h = mmcv.rescale_size((rgb_img_w, rgb_img_h), self.scale)
            sk_new_w, sk_new_h =  rgb_new_w, rgb_new_h

        else:
            rgb_new_w, rgb_new_h = self.scale
            sk_new_w, sk_new_h =  rgb_new_w, rgb_new_h

        self.rgb_scale_factor = np.array([rgb_new_w / rgb_img_w, rgb_new_h / rgb_img_h],
                                     dtype=np.float32)
        self.sk_scale_factor = np.array([sk_new_w / sk_img_w, sk_new_h / sk_img_h],
                                     dtype=np.float32)

        results['rgb_img_shape'] = (rgb_new_h, rgb_new_w)
        results['sk_img_shape'] = (sk_new_h, sk_new_w)


        results['keep_ratio'] = self.keep_ratio
        results['rgb_scale_factor'] = results['rgb_scale_factor'] * self.rgb_scale_factor
        results['sk_scale_factor'] = results['sk_scale_factor'] * self.sk_scale_factor

        if not self.lazy:
            if 'rgb_imgs' in results:
                results['rgb_imgs'] = self._resize_imgs(results['rgb_imgs'], rgb_new_w,
                                                    rgb_new_h)
                # print(1111)
            if 'sk_imgs' in results:
                results['sk_imgs'] = self._resize_imgs(results['sk_imgs'], sk_new_w,
                                                    sk_new_h)
                # print(1111)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation


        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str

class  Video_Sk_RandomCrop(BaseTransform):
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Static method for cropping keypoint."""
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Static method for cropping images."""
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def transform(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str

@TRANSFORMS.register_module()
class  Video_Sk_RandomResizedCrop(Video_Sk_RandomCrop):
    ###########——————————————————————》 如果rgb与sk的比例不同，则不能执行此函数，将会打断二者比例联系 《——————————————————————##############
    ###########——————————————————————》 如果rgb与sk的比例不同，则不能执行此函数，将会打断二者比例联系 《——————————————————————##############
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmengine.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def transform(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        ########### 此处有随机函数，所只针对rgb进行。##############
        img_h, img_w = results['rgb_img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['rgb_img_shape'] = (new_h, new_w)
        results['sk_img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'rgb_imgs' in results:
                results['rgb_imgs'] = self._crop_imgs(results['rgb_imgs'], crop_bbox)
            if 'sk_imgs' in results:
                results['sk_imgs'] = self._crop_imgs(results['sk_imgs'], crop_bbox)
            # print(11111)

        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str

@TRANSFORMS.register_module()
class Video_Sk_CenterCrop(Video_Sk_RandomCrop):
    ##### 先做同一比例
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "lazy" and
    "img_shape". Required keys in "lazy" is "crop_bbox", added or modified key
    is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmengine.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        img_h, img_w = results['rgb_img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['rgb_img_shape'] = (new_h, new_w)
        results['sk_img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'rgb_imgs' in results:
                results['rgb_imgs'] = self._crop_imgs(results['rgb_imgs'], crop_bbox)
            if 'sk_imgs' in results:
                results['sk_imgs'] = self._crop_imgs(results['sk_imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str

@TRANSFORMS.register_module()
class  Video_Sk_Flip(BaseTransform):
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". The Flip augmentation should be placed
    after any cropping / reshaping augmentations, to make sure crop_quadruple
    is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy

    def _flip_imgs(self, imgs, ):
        """Utility function for flipping images."""
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        return imgs

    def transform(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            if flip:
                if 'rgb_imgs' in results:
                    results['rgb_imgs'] = self._flip_imgs(results['rgb_imgs'],
                                                      )
                if 'sk_imgs' in results:
                    results['sk_imgs'] = self._flip_imgs(results['sk_imgs'],
                                                      )
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction



        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


@TRANSFORMS.register_module()
class  Video_Sk_FormatShape(BaseTransform):
    """Format final imgs shape to the given input_format.

    Required keys:

        - imgs (optional)
        - heatmap_imgs (optional)
        - modality (optional)
        - num_clips
        - clip_len

    Modified Keys:

        - imgs

    Added Keys:

        - input_shape
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
                'NTCHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['rgb_imgs'], np.ndarray):
            results['rgb_imgs'] = np.array(results['rgb_imgs'])
        if not isinstance(results['sk_imgs'], np.ndarray):
            results['sk_imgs'] = np.array(results['sk_imgs'])
        imgs = np.concatenate((results['rgb_imgs'], results['sk_imgs']), axis=0)

        results['imgs'] =imgs
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NTCHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs[np.newaxis, :]
                # print(11111)
                # N x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 4, 2,3))
                # N_crops x N_clips x C x T x H x W
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            imgs = imgs[np.newaxis, :]
            if 'modality' in results and results['modality'] == 'Flow':
                clip_len = results['clip_len']
                imgs = imgs.reshape((-1, clip_len * imgs.shape[1]) +
                                    imgs.shape[2:])
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str