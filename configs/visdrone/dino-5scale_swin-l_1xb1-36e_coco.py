_base_ = '../dino/dino-5scale_swin-l_8xb2-12e_coco.py'

# dataset settings
data_root = '/home/yang/data/smoke-fire-person-dataset/person/VisDrone'
dataset_type = 'CocoDataset'

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
        ))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/val.json',
        data_prefix=dict(img='val/images'),
        test_mode=True,
        ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/val/val.json',
    metric='bbox',
    format_only=False,
    )
test_evaluator = val_evaluator

max_epochs = 36
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1)
]
