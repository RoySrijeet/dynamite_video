# Downloading Pre-trained weights

Save the downloaded weights in the `weights` directory, and specify the correct path in MODEL.WEIGHTS in experiment configuration.

## Downloading DynaMITe weights

DynaMITe weights can be downloaded from the [official git repository](https://github.com/amitrana001/DynaMITe).

## Downloading backbone weights

We follow [Mask2Former guidlines](https://github.com/facebookresearch/Mask2Former/blob/main/tools/README.md) to download and convert the pretrained models for backbones.

The `weights/mask2former_weight_tools` directory contains the above-mentioned tools.

```
# Swin-T
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
python mask2former_weight_tools/convert-pretrained-swin-model-to-d2.py swin_tiny_patch4_window7_224.pth swin_tiny_patch4_window7_224.pkl

# Swin-L
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
python mask2former_weight_tools/convert-pretrained-swin-model-to-d2.py swin_large_patch4_window12_384_22k.pth swin_large_patch4_window12_384_22k.pkl
```