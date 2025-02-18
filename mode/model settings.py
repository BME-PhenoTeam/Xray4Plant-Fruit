# model settings

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNet3Plus',
        in_channels=3,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
        Attention_choose=False,        
        Original_encoder=False        
        # type='UNet3Plus',
        # skip_ch=64,
        # encoder='RepLKNet',
        # channels=[3, 128, 256, 512, 1024],
        # dropout=0.3,
        # fast_up=True
    ),
    decode_head=dict(
        type='Unet3_plus_Head',
        in_channels=64,
        channels=64,
        aux_losses=4,
        skip_ch=64,
        use_cgm=False,
        transpose_final=True,
        loss_type='focal',
        aux_weight=0.4,
        process_input=True,
        num_classes=5,
        backbone_channels=[3, 64, 128, 256, 512, 1024],
        norm_cfg=norm_cfg,
        align_corners=False
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))

