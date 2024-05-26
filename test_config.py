_base_ = ['D:\Mxd\mmaction\mmaction2\configs\_base_\default_runtime.py']

# model settings
num_frames = 8
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='UniFormerV2',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[8, 9, 10, 11],
        n_layers=4,
        n_dim=768,
        n_head=12,
        mlp_factor=4.,
        drop_path_rate=0.,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
        clip_pretrained=True,
        pretrained='ViT-B/16'),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.5,
        num_classes=4,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

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
    dict(type='Video_Sk_FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='Video_Sk_SampleFrames', clip_len=num_frames, num_clips=1),
    dict(type='Video_Sk_RawFrameDecode'),
    dict(type='Video_Sk_Resize', scale=(-1, 224)),
    dict(type='Video_Sk_CenterCrop', crop_size=224),
    dict(type='Video_Sk_FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')


base_lr = 1e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2))

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
        end=55,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=10))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
# auto_scale_lr = dict(enable=True, base_batch_size=256)
