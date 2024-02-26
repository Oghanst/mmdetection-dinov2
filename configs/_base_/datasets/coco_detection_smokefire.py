# dataset settings
dataset_type = 'CocoDataset_Smokefire'
data_root = '/home/yang/data/smoke-fire-person-dataset/smoke-fire/smokefire_v4' # 数据的根路径。

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # dict(
    #     type='RandomChoice',
    #     transforms=[
    #         [
    #             dict(
    #                 type='RandomChoiceResize',
    #                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                         (736, 1333), (768, 1333), (800, 1333)],
    #                 keep_ratio=True)
    #         ],
    #         [
    #             dict(
    #                 type='RandomChoiceResize',
    #                 # The radio of all image in train dataset < 7
    #                 # follow the original implement
    #                 scales=[(400, 4200), (500, 4200), (600, 4200)],
    #                 keep_ratio=True),
    #             dict(
    #                 type='RandomCrop',
    #                 crop_type='absolute_range',
    #                 crop_size=(384, 600),
    #                 allow_negative_crop=True),
    #             dict(
    #                 type='RandomChoiceResize',
    #                 scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                         (736, 1333), (768, 1333), (800, 1333)],
    #                 keep_ratio=True)
    #         ]
    #     ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/images'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/test.json',
        data_prefix=dict(img='test/images'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/test/test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator