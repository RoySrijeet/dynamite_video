from .registry import MODEL_REGISTRY

from detectron2.layers import ShapeSpec

def build_backbone(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = MODEL_REGISTRY.get(backbone_name)(cfg, input_shape)
    return backbone


# def build_sem_seg_head(cfg, input_shape):
#     """
#     Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
#     """
#     name = cfg.MODEL.SEM_SEG_HEAD.NAME
#     return MODEL_REGISTRY.get(name)(cfg, input_shape)


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = MODEL_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


def build_interactive_transformer(cfg, in_channels):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.INTERACTIVE_TRANSFORMER_NAME
    return MODEL_REGISTRY.get(name)(cfg, in_channels)