_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,
    backbone=dict(
        type='FocalNet',
        embed_dim=96,
        depths=[4, 4, 28, 4],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        focal_windows=[9,9,9,9],
        focal_levels=[3,3,3,3],
        use_conv_embed=False, 
        use_layerscale=False, 
    ),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))
