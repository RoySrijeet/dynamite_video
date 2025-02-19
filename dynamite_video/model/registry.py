from .backbones import SwinTransformer
from .pixel_decoder import BasePixelDecoder, TransformerEncoderPixelDecoder, MSDeformAttnPixelDecoder
from .interactive_transformer import DynamiteInteractiveTransformer

MODEL_REGISTRY = {
    "SwinTransformer": SwinTransformer,
    "BasePixelDecoder": BasePixelDecoder,
    "TransformerEncoderPixelDecoder": TransformerEncoderPixelDecoder,
    "MSDeformAttnPixelDecoder": MSDeformAttnPixelDecoder,
    "DynamiteInteractiveTransformer": DynamiteInteractiveTransformer,
}