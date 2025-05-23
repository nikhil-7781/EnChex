# --------------------------------------------------------
# Swin Transformer (Hybrid XAI extension)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, extended for hybrid/XAI by user
# --------------------------------------------------------

from .swin_transformer import SwinTransformer

# ---- import your hybrid model ----
from .hybrid_swin_cnn import HybridSwinCNN  # Make sure this file exists and class is implemented

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            num_mlp_heads=config.NIH.num_mlp_heads
        )
    elif model_type == 'hybrid_swin_cnn':
        model = HybridSwinCNN(
            config=config,
            cnn_backbone=getattr(config.MODEL, "CNN_BACKBONE", "resnet50"),
            pretrained=True,
            num_classes=config.MODEL.NUM_CLASSES
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model
