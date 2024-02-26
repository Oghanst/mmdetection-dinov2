# ----- wanbd_runtime -----
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',
                    interval=4, 
                    save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

#TODO: 后面需要修改自定义这个name
vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         project='pure-seg',
                         name = f'dino_dinov2_visdrone',
                         group='dino_dinov2',
                         resume=True))
                ]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False