import cv2
import math
import torch
import numpy as np
import imgaug.augmenters as iaa
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from collections import defaultdict
from typing import List, Tuple, Union, Optional, Dict
from detectron2.data import transforms as T


def scale_and_normalize_images(images, means, scales, invert_channels, normalize_to_unit_scale):
    """
    Adapted from https://github.com/Ali2500/TarViS
    Scales and normalizes images
    :param images: tensor(T, C, H, W)
    :param means: list(float)
    :param scales: list(float)
    :param invert_channels: bool
    :param normalize_to_unit_scale: bool
    :return: tensor(T, C, H, W)
    """
    means = torch.as_tensor(means, dtype=torch.float32)[None, :, None, None]  # [1, 3, 1, 1]
    scales = torch.as_tensor(scales, dtype=torch.float32)[None, :, None, None]  # [1. 3. 1. 1]
    if normalize_to_unit_scale:
        images = images / 255.

    images = (images - means) / scales
    if invert_channels:
        return images.flip(dims=[1])
    else:
        return images


def condense_mask(mask: Union[Tensor, np.ndarray], dtype: Optional[Union[np.dtype, torch.dtype]] = None):
    """
    Adapted from https://github.com/Ali2500/TarViS
    Condense a one-hot mask into a mask array where pixel value denotes instance ID
    :param mask: Numpy or torch array of shape [N, H, W]
    :param dtype: dtype of output mask
    :return: Numpy or torch array of shape [H, W]
    """
    height, width = mask.shape[1:]
    if torch.is_tensor(mask):
        dtype = torch.long if dtype is None else dtype
        condensed_mask = torch.zeros(height, width, dtype=dtype, device=mask.device)
        where_fn = torch.where
    else:
        dtype = np.int32 if dtype is None else dtype
        condensed_mask = np.zeros((height, width), dtype)
        where_fn = np.where

    for iid, mask_per_channel in enumerate(mask, 1):
        condensed_mask = where_fn(mask_per_channel, iid, condensed_mask)

    return condensed_mask


def expand_mask(mask: Union[Tensor, np.ndarray], instance_ids: Optional[List[int]] = None):
    """
    Expand a condensed mask into a one-hot mask array
    :param mask: Numpy or torch array of shape [H, W] where pixel values denote instace/class ID
    :param instance_ids: Optional list of instance IDs. If not provided, all values in the given mask will be used.
    :return: Numpy or torch array of shape [N, H, W]
    """
    if instance_ids is None:
        if torch.is_tensor(mask):
            instance_ids = mask.unique()
        else:
            instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0].tolist()

    expanded_mask = [mask == iid for iid in instance_ids]

    if torch.is_tensor(mask):
        expanded_mask = torch.stack(expanded_mask, 0).bool()
    else:
        expanded_mask = np.stack(expanded_mask).astype(bool)

    return expanded_mask


def compute_resized_dims(height: int, width: int, min_dim: int, max_dim: int):
    """
    Compute new dimensions for resizing images. Resizes the image such 
    that the smaller side is equal to min_dim. Maintains the aspect ratio.
    
    Args:
        height: original height
        width: original width
        min_dim: minimum dimension
        max_dim: maximum dimension
    """
    dims = (height, width)
    lower_size = float(min(dims))
    higher_size = float(max(dims))

    if isinstance(min_dim, (list, tuple)):
        min_dim = min_dim[torch.randint(len(min_dim), (1,)).item()]
    else:
        min_dim = min_dim

    scale_factor = min_dim / lower_size
    if (higher_size * scale_factor) > max_dim:
        scale_factor = max_dim / higher_size

    new_height, new_width = round(scale_factor * height), round(scale_factor * width)

    return new_height, new_width


def resize_images(images, new_height, new_width):
    """
    Resize images to new dimensions
    
    Args:
        images: tensor or np.ndarray of shape [B, H, W, C]
        new_height: target height
        new_width: target width
    """
    # resize image
    if torch.is_tensor(images):
        images = rearrange(images, "B H W C -> B C H W")
        images = F.interpolate(images, (new_height, new_width), mode='bilinear', align_corners=False)
        images = rearrange(images, "B C H W -> B H W C")
    else:   # faster with np
        assert isinstance(images, np.ndarray), f"Unexpected image type: {type(images)}, expected np.ndarray or torch.Tensor"
        images = np.stack([cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR) for im in images])

    return images


def resize_masks(masks, new_height, new_width, binary=False):
    """
    Resize masks to new dimensions

    Args:
        masks: tensor or np.ndarray of shape [N, T, H, W] (for binary masks) 
            or [T, H, W] for semantic masks
        new_height: target height
        new_width: target width
        binary: whether the masks are binary or semantic
    """
    if not binary:
        # resizing semantic masks
        
        assert masks.ndim == 3, f"Expected shape of semantic mask [T, H, W], got {masks.ndim} dimensions instead."
        
        dtype = masks.dtype
        if torch.is_tensor(masks):
            # NOTE - interpolation mode is set to 'nearest' instead of 'bilinear'
            resized_mask = F.interpolate(masks.float(), (new_height, new_width), mode='nearest', align_corners=False)
            resized_mask = (resized_mask > 0.5).astype(dtype)

            return resized_mask
        else:
            assert isinstance(masks, np.ndarray), f"Unexpected mask type: {type(masks)}, expected np.ndarray or torch.Tensor"
            
            # NOTE - interpolation mode is set to 'INTER_NEAREST' instead of 'INTER_LINEAR'
            resized_mask = np.stack([cv2.resize(m, (new_width, new_height), interpolation=cv2.INTER_NEAREST) for m in masks])
        
            return resized_mask
    
    else:
        # resizing binary masks
        assert masks.ndim == 4, f"Expected shape of semantic mask [T, N, H, W], got {masks.ndim} dimensions instead."
        dtype = masks.dtype
        if torch.is_tensor(masks):
            # NOTE - bilinear interpolation with thresholding
            resized_masks = F.interpolate(masks.float(), (new_height, new_width), mode='bilinear', align_corners=False)
            resized_masks = (resized_masks > 0.5).astype(dtype)
        
            return resized_mask
        else:
            assert isinstance(masks, np.ndarray), f"Unexpected mask type: {type(masks)}, expected np.ndarray or torch.Tensor"
            
            B, N = masks.shape[:2]
            resized_masks = np.reshape(masks, (-1, *masks.shape[2:]))
            
            # NOTE - bilinear interpolation with thresholding
            resized_masks = np.stack([
                (cv2.resize(m.astype(np.float32), (new_width, new_height),
                            interpolation=cv2.INTER_LINEAR) > 0.5).astype(dtype)
                for m in resized_masks
            ])
            resized_masks = np.reshape(resized_masks, (B, N, *resized_masks.shape[1:]))
        
            return resized_masks
        
def apply_resizer(
        images: np.ndarray, 
        binary_masks: np.ndarray,
        # semantic_masks: np.ndarray,
        mode: str,
        min_dim: int,
        max_dim: int,
    ):
        """
        Resize video frames to a specified resolution .Shortest edge of each frame 
        is reduced to the target size.

        Args:
            images: [T, H, W, 3]
            binary_masks: [N, T, H, W]
            mode: currently only supports "min_dim" which corresponds to resizing the
                shortest edge
            min_dim: resize shorter edge to this size
            max_dim: maximum dimension cut-off for the longer edge
        """
        ALLOWED_MODES = ["min_dim"]
        assert mode in ALLOWED_MODES, f"Desired resize mode {mode} is not available. \
            Choose from {ALLOWED_MODES}"

        if mode == "none":
            return images, binary_masks

        # compute target resolution
        new_height, new_width = compute_resized_dims(
            *images.shape[1:3], 
            min_dim, 
            max_dim,
        )

        images = resize_images(images, new_height, new_width)

        binary_masks = resize_masks(binary_masks, new_height, new_width, binary=True)

        return images, binary_masks



def apply_color_augmentation(images: List[np.ndarray]):
        """
        Apply same color augmentation to all frames

        Args:
            images: list of RGB images [H, W, 3]
        """
        color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-25, 25))
        ])
        det_augmenter = color_augmenter.to_deterministic()
        return [det_augmenter(image=img) for img in images]


def apply_random_horizontal_flip(
        images: np.ndarray, 
        binary_masks: np.ndarray,
        flip_axis: str,
        prob: float
    ):
        """
        Apply random horizontal flips
        
        Args:
            images: [T, H, W, 3]
            binary_masks: [N, T, H, W]
            flip_axis: whether to flip horizontal or vertical.
                Currently only "horizontal" flips are supported
            prob: apply flip with this probability
        """
        assert images.ndim == 4 and binary_masks.ndim == 4
        
        # only horizontal flips
        assert flip_axis == "horizontal", f"Only 'horizontal' flips are allowed, {flip_axis} is not allowed!"

        if torch.rand(1) < prob:
            # flip along width
            images = np.flip(images, 2).copy()
            binary_masks = np.flip(binary_masks, 3).copy()

        return images, binary_masks


def apply_random_crop(
        images: np.ndarray, 
        binary_masks: np.ndarray, 
        instance_ids,
        crop_size,
        MIN_MASK_AREA,
    ):
        """
        Apply random horizontal flips
        
        Args:
            images: [T, H, W, 3]
            binary_masks: [N, T, H, W]
            instance_ids: list of instances in the clip
            crop_size: randomly crop area of this dimensions
            MIN_MASK_AREA: minimum area threshold
        """
        assert images.ndim == 4 and binary_masks.ndim == 4 # and semantic_masks.ndim == 3
        crop_size = (crop_size, crop_size)

        # input dims
        input_size = images.shape[1:3]
        # crop offset limits
        max_offset = np.subtract(input_size, crop_size)
        max_offset = np.maximum(max_offset, 0)

        attempt = 0
        while True:
            
            # randomly pick a crop offset
            offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
            offset = np.round(offset).astype(int)
            # apply crop
            cropped_images = images[..., offset[0]:offset[0]+crop_size[0], offset[1]:offset[1]+crop_size[1], :]
            cropped_binary_masks = binary_masks[:, :, offset[0]:offset[0]+crop_size[0], offset[1]:offset[1]+crop_size[1]]
            
            # ensure remaining mask sizes are valid
            valid = True
            avg_area = 0
            for fr_msk in cropped_binary_masks:
                avg_area += fr_msk.sum()
                for _msk in fr_msk:
                    if _msk.sum() > 0 and _msk.sum() < MIN_MASK_AREA:
                        # existing mask is smaller than threshold
                        valid = False
                        break
            # empty mask
            if avg_area==0 or avg_area // cropped_binary_masks.shape[0] < MIN_MASK_AREA:
                valid = False
            if valid:
                break
            
            attempt += 1
            # if can't get a valid crop, resize to target crop dims
            if attempt >=3:
                cropped_images = resize_images(images, crop_size[0], crop_size[1])
                cropped_binary_masks = resize_masks(binary_masks, crop_size[0], crop_size[1], binary=True)
                break

        cropped_size = cropped_images.shape[1:3]
        pad_size = np.subtract(crop_size, cropped_size)
        pad_size = np.maximum(pad_size, 0)
        # account for applied mask
        padding_mask = np.ones(cropped_binary_masks.shape[2:])
        if pad_size.sum() > 0:
            # image
            im_padding = ((0,0), (0, pad_size[0]), (0, pad_size[1]), (0,0))
            cropped_images = np.pad(cropped_images, im_padding, mode='constant', constant_values=128.0)

            # binary masks
            binary_mask_padding = ((0,0), (0,0), (0, pad_size[0]), (0, pad_size[1]))
            cropped_binary_masks = np.pad(cropped_binary_masks, binary_mask_padding, mode='constant', constant_values=0)

            padding = ((0, pad_size[0]), (0, pad_size[1]))
            padding_mask = np.pad(padding_mask, padding, mode='constant', constant_values=0)

        padding_mask = np.logical_not(padding_mask)
        
        # generate semantic map
        semantic_masks = []
        # record which frame contains which instances
        frame_instance_occupancy = defaultdict(list)
        for fr_idx, fr_mask in enumerate(cropped_binary_masks):
            semantic_mask_fr = np.zeros(fr_mask[0].shape)
            for idx, inst_id in enumerate(instance_ids):
                if fr_mask[idx].sum() > 0:
                    frame_instance_occupancy[inst_id].append(fr_idx)
                    semantic_mask_fr[np.where(fr_mask[idx]==1)] = inst_id
            semantic_masks.append(semantic_mask_fr)
        semantic_masks = np.stack(semantic_masks).astype('uint8')

        return cropped_images, cropped_binary_masks, semantic_masks, padding_mask, frame_instance_occupancy

