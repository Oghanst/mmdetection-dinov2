_base_ = [
    '../_base_/datasets/coco_detection_visdrone.py', '../_base_/wandb_runtime.py'
]
# How to run it?
# 1. immport mmpretrain
# export PYTHONPATH=$PWD:$PYTHONPATH
# CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/dino/dino-4scale_dinov2-base_8xb2-12e_visdrone.py --work-dir work_dirs/dino-4scale_dinov2-base_8xb2-12e_visdrone
# CUDA_VISIBLE_DEVICES=2 nohup python tools/train.py configs/dino/dino-4scale_dinov2-base_8xb2-12e_visdrone.py --work-dir work_dirs/dino-4scale_dinov2-base_8xb2-12e_visdrone > work_dirs/dino_dinov2_base.log 2>&1 &
# 3. run test.py
# CUDA_VISIBLE_DEVICES=1 python tools/test.py configs/dino/dino-4scale_dinov2-base_8xb2-12e_visdrone.py work_dirs/dino-4scale_dinov2-base_8xb2-12e_visdrone/best_coco_bbox_mAP_epoch_26.pth  --show-dir work_dirs/dino-4scale_dinov2-base_8xb2-12e_visdrone/show_dir


# global configs
learning_rate = 0.01
max_epochs = 100
num_levels = 4
crop_size = (1024,1024)
batch_size = 2
pretrained = 'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth'
model = dict(
    type='DINO',
    num_queries=100,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch = 'base',
        patch_size=14,
        out_type = "featmap",
        out_indices=(0, 1, 2, 3),
        frozen_stages = 1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
            prefix='backbone.',
        )
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768]*4,
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),

    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=1,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),



    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries

    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

train_dataloader = dict(
    batch_size=batch_size,
)
val_dataloader = dict(
    batch_size=batch_size,
)

# vis
vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='WandbVisBackend',
                     init_kwargs=dict(
                         project='Xin-Dalu',
                         name = f'dino_{num_levels}scale_dinov2_base_frozen_visdrone_lr{learning_rate}_{max_epochs}e',
                         group='dino_dinov2_base',
                         resume=True))
                ]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)


# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=learning_rate,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=learning_rate, by_epoch=False, begin=0, end=1),
    dict(
        type='CosineAnnealingLR', 
        by_epoch=True, 
        begin=1,
        T_max=max_epochs,
        end=max_epochs,
    )
]
