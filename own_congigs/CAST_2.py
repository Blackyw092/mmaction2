_base_ = ['_base_/default_runtime.py']
import wandb

# model settings
num_frames = 8
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='STCrossTransformer_2',),
    cls_head=dict(
        type='UniFormerHead',
        num_classes=51,
        in_channels=768,
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCHW')
)

# dataset settings
dataset_type = 'Video_Sk_Dataset'
data_root = 'data'
data_root_val = 'data'
ann_file_train = 'D:\Mxd\mmaction\mmaction2\hmdb51-mix\\train_frame.txt'
ann_file_val = 'D:\Mxd\mmaction\mmaction2\hmdb51-mix\\val_frame.txt'

train_pipeline = [
    dict(type='Video_Sk_SampleFrames', clip_len=1, num_clips=num_frames),
    dict(type='Video_Sk_RawFrameDecode'),
    dict(type='Video_Sk_Resize', scale=(-1, 256)),
    dict(type='Video_Sk_RandomResizedCrop'),
    dict(type='Video_Sk_Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Video_Sk_Flip', flip_ratio=0.5),
    dict(type='Video_Sk_FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='Video_Sk_SampleFrames', clip_len=1, num_clips=num_frames),
    dict(type='Video_Sk_RawFrameDecode'),
    dict(type='Video_Sk_Resize', scale=(-1, 256)),
    dict(type='Video_Sk_CenterCrop', crop_size=224),
    dict(type='Video_Sk_FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')

visualizer = dict(vis_backends=[dict(type='WandbVisBackend')])

base_lr = 5.e-6
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',lr=base_lr, betas=(0.9, 0.98), weight_decay=0.02,eps=1e-8),
)
    # clip_grad=dict(max_norm=20, norm_type=2)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min_ratio=0.1,
        by_epoch=True,
        begin=5,
        end=50,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5),)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
# auto_scale_lr = dict(enable=True, base_batch_size=256)
# resume = True
