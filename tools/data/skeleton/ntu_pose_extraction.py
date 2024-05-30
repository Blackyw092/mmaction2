# Copyright (c) OpenMMLab. All rights reserved.
import abc
import argparse
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory

import mmengine
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import os

from mmaction.apis import detection_inference, pose_inference
from mmaction.utils import frame_extract

args = abc.abstractproperty()
args.det_config = 'D:\mmaction\mmaction2\demo\demo_configs\\faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'  # noqa: E501
args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_score_thr = 0.5
args.pose_config = 'D:\mmaction\mmaction2\demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'  # noqa: E501
args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):

    def inside(box0, box1, threshold=0.8):
        return intersection(box0, box1) / area(box0) > threshold

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    threshold = 0.95

    def thre_bbox(bboxes, threshold=threshold):
        shape = [sum(bbox[:, -1] > threshold) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    iou_thre = 0.6
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets


def drop_tracklet(tracklet):
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    dists = {}
    for k, v in tracklet.items():
        bboxes = np.stack([x[1] for x in v])
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.
        c_x -= 480
        c_y -= 270
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)
    return dists


def tracklet2bbox(track, num_frame):
    # assign_prev
    bbox = np.zeros((num_frame, 5))
    trackd = {}
    for k, v in track:
        bbox[k] = v
        trackd[k] = v
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:
            mind = np.Inf
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
            bbox[i] = bbox[k]
    return bbox


def tracklets2bbox(tracklet, num_frame):
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            dist_thre = 2 * dists[i]
            break

    dist_thre = max(50, dist_thre)

    bbox = np.zeros((num_frame, 5))
    bboxd = {}
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:
                    bbox[k] = v
                    bboxd[k] = v
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k
            bbox[idx] = bboxd[mink]
    return bad, bbox[:, None, :]


def bboxes2bbox(bbox, num_frame):
    ret = np.zeros((num_frame, 2, 5))
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            ret[t, :item.shape[0]] = item
        else:
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]
    for t in range(num_frame):
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]
        elif ret[t, 1, -1] <= 0.01:
            if t:
                if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                    if iou(ret[t, 0], ret[t - 1, 0]) > iou(
                            ret[t, 0], ret[t - 1, 1]):
                        ret[t, 1] = ret[t - 1, 1]
                    else:
                        ret[t, 1] = ret[t - 1, 0]
    return ret


'''
您可能唯一需要更改的一件事是：由于ntu_pose_extraction.py是专门为 NTU 视频的姿势提取而开发的，
-因此在使用此脚本从自定义视频数据集中提取姿势时，您可以跳过ntu_det_postproc步骤。
'''
def ntu_det_postproc(vid, det_results):
    """
    对NTU视频数据集的检测结果进行后处理。

    参数:
    vid: 字符串，视频的文件路径或标识符，用于提取视频标签。
    det_results: 列表，包含视频中检测到的物体框的列表。

    返回值:
    根据视频的难度和人数，返回优化后的物体框或跟踪轨迹。

    """
    # 移除检测结果中的重复项
    det_results = [removedup(x) for x in det_results]
    # 从视频标识符中提取标签号
    label = int(vid.split('/')[-1].split('A')[1][:3])
    # 定义动作类别，决定是单人还是双人动作
    mpaction = list(range(50, 61)) + list(range(106, 121))
    n_person = 2 if label in mpaction else 1
    # 判断是否为简单示例
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        # 简单示例，直接返回优化后的物体框
        print('\nEasy Example')
        return bboxes

    # 转换检测结果为跟踪轨迹
    tracklets = bbox2tracklet(det_results)
    # 移除不良的跟踪轨迹
    tracklets = drop_tracklet(tracklets)

    # 打印难例信息，并根据人数处理跟踪轨迹
    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')
    if n_person == 1:
        # 单人动作处理逻辑
        if len(tracklets) == 1:
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results
    # 双人动作处理逻辑
    if len(tracklets) <= 2:
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        # 处理多人情况，返回合并后的物体框
        return bboxes2bbox(det_results, len(det_results))



def pose_inference_with_align(args, frame_paths, det_results):
    """
    使用对齐方式进行姿态推断的函数。

    参数:
    args: 姿态推断所需的配置参数，包括pose_config（姿态模型配置文件路径},
          pose_checkpoint（姿态模型的检查点路径），device（运行设备）等。
    frame_paths: 图像帧的路径列表。
    det_results: 检测结果列表，每个元素包含一个帧的检测到的人体边界框信息。

    返回:
    keypoints: 对齐后，每个人体关节点在所有帧中的坐标。
    scores: 对齐后，每个人体关节点在所有帧中的得分。
    """

    # 过滤掉没有检测到人体边界框的帧
    det_results = [
        frm_dets for frm_dets in det_results if frm_dets.shape[0] > 0
    ]

    # 进行姿态推断，并获取每个人体关节点的坐标和得分
    pose_results, _ = pose_inference(args.pose_config, args.pose_checkpoint,
                                     frame_paths, det_results, args.device)

    # 计算所有帧中出现的最大人数和每个关节点的数目
    num_persons = max([pose['keypoints'].shape[0] for pose in pose_results])
    num_points = pose_results[0]['keypoints'].shape[1]
    num_frames = len(pose_results)

    # 初始化用于存储对齐后结果的数组
    keypoints = np.zeros((num_persons, num_frames, num_points, 2),
                         dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    # 填充对齐后结果的数组
    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
            scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

    return keypoints, scores


def ntu_pose_extraction(vid,filename, skip_postproc=False):
    """
    从视频中提取人体关节点信息。

    参数:
    vid: 输入视频的路径。
    skip_postproc: 是否跳过后处理步骤。默认为False，即执行后处理。

    返回值:
    anno: 包含关节点信息的字典，包括关节点位置、关节点得分、帧目录、图像尺寸等。
    """
    # cap = cv2.VideoCapture(vid)
    # ret, frame = cap.read()
    # frame_size = frame.shape[:2]
    # print(frame_size)
    # 创建临时目录存储提取的帧
    tmp_dir = TemporaryDirectory()
    frame_paths, _ = frame_extract(vid, out_dir=tmp_dir.name)  # 提取视频帧并获取路径
    image = Image.open(frame_paths[0])
    width, height = image.size
    image.close()

    # 对视频帧进行物体检测
    det_results, _ = detection_inference(
        args.det_config,
        args.det_checkpoint,
        frame_paths,
        args.det_score_thr,
        device=args.device,
        with_score=True)

    # 如果未跳过后处理，则对检测结果进行后处理
    # if not skip_postproc:
    #     det_results = ntu_det_postproc(vid, det_results)  # 对检测结果进行后处理

    anno = dict()


    # 进行关节点定位，同时进行对齐处理
    keypoints, scores = pose_inference_with_align(args, frame_paths,
                                                  det_results)
    # 填充关节点信息到anno字典
    anno['keypoint'] = keypoints
    anno['keypoint_score'] = scores
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]  # 提取视频文件名的基础部分
    anno['img_shape'] = (1080, 1920)  # 图像的目标尺寸
    anno['original_shape'] = (height, width)  # 图像的原始尺寸
    anno['total_frames'] = keypoints.shape[1]  # 总帧数
    # print(anno)
    # 从视频文件名中提取标签，并转换为索引

    # print(osp.basename(vid).split('-')[1])
    anno['label'] = str(filename)
    # 清理临时目录
    tmp_dir.cleanup()
    # print(anno)

    return anno



def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    # parser.add_argument('video', type=str, help='source video')
    # parser.add_argument('output', type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    work_dir = 'D:\mmaction\mmaction2\hmdb51'
    filename_list = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    type_number = len(filename_list)
    global_args = parse_args()
    for process_index in range(type_number):
        filename = filename_list[process_index]
        p = Path(work_dir + '/' + filename)
        # print(p)
        for files in os.listdir(p):
            vid_path = Path(work_dir + '/' + filename +'/' + files)
            pkl_path = Path(work_dir + '/' + filename +'/' + files.split('.')[0]+'.pkl')
            # print(pkl_path)

                # if file.endswith('.avi'):
                #     args = dict()
                #     args.video = dirname + '/' + file
                #     args.output = dirname + '/' + file.split('.')[0] + '.pkl'
                #     args.skip_postproc = global_args.skip_postproc
            args.device = global_args.device
            args.video = str(vid_path)
            args.output = str(pkl_path)
            args.skip_postproc = global_args.skip_postproc
            anno = ntu_pose_extraction(args.video, filename,args.skip_postproc)
            mmengine.dump(anno, args.output)
