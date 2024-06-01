default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, ignore_last=False,),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True,
                     custom_cfg=[
                        dict(data_src='top1_acc',
                             log_name='top1_acc_mean',
                             method_name='mean',
                             window_size=10
                                 ),
                         dict(data_src='loss',
                              log_name='loss_mean',
                              method_name='mean',
                              window_size=10
                              ),
                         dict(data_src='val/loss',
                              method_name='mean',
                              window_size=10
                              ),

                     ])  # 统计窗口：全局)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
