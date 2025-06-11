import cv2
import imgaug.augmenters as iaa
import numpy as np
import pycocotools.mask as mt
import random
import torch
import torch.nn.functional as F

from collections import OrderedDict
from einops import rearrange
from torch import Tensor
from typing import List, Tuple, Optional


def decode_mask(encoded_mask, size=None):
    """
    Decode RLE mask into `np.ndarray`

    Args:
        encoded_mask: RLE mask
        size: mask dimensions
    
    Returns:
        `np.ndarray` of dimensions `size`
    """
    if size is None:
        assert isinstance(encoded_mask, dict)
        assert 'counts' in encoded_mask.keys()
        assert 'size' in encoded_mask.keys()
        return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)

    if isinstance(encoded_mask, list):  # polygons
        encoded_mask = {
            "counts": encoded_mask,
            "size": size,
        }
        encoded_mask = mt.frPyObjects(encoded_mask, size[0], size[1])
    
    else:  # RLE mask
        assert isinstance(encoded_mask, str), f"Unexpected encoded mask type: {type(encoded_mask)}"
        encoded_mask = {
            "counts": encoded_mask.encode("utf-8"),
            "size": size
        }
    
    return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)


def serialize_object_ids(orig_ids):
    """
    Serialize object IDs. IDs are 1-indexed to avoid conflict in semantic mask
    with background pixels (0)

    Args:
        orig_ids: original object IDs, potentially non-sequential

    Returns:
        orig_to_serial_id: mapping from original IDs to sequential IDs
        serial_to_orig_id: mapping from sequential IDs to original IDs
    """
    
    orig_ids = sorted(orig_ids)
    serial_ids = [i for i in range(1, len(orig_ids)+1)]
    serial_to_orig_id = OrderedDict(zip(serial_ids, orig_ids))
    orig_to_serial_id = OrderedDict(zip(orig_ids, serial_ids))
    return orig_to_serial_id, serial_to_orig_id


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
    return np.stack([det_augmenter(image=img) for img in images])


def resize_images(images, new_height: int, new_width: int):
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


def resize_masks(masks, new_height:int , new_width: int, binary: bool=False):
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

def apply_random_flip(
        images: np.ndarray, 
        binary_masks: np.ndarray,
        ignore_masks: np.ndarray|None,
        axis: str="horizontal",
        prob: float=0.5,
):
    """
    Apply flips with a random probability. Either all frames in a clip are flipped or none.
    
    Args:
        images: [T, H, W, 3]
        binary_masks: [T, N, H, W]
        ignore_masks: [T, H, W]
        flip_axis: currently only "horizontal" flips are supported
        prob: apply flip with this probability
    """
    assert images.ndim == 4 and binary_masks.ndim == 4
    assert axis == "horizontal", f"Only 'horizontal' flips are allowed, {axis} is not allowed!"

    if torch.rand(1) < prob:
        # flip along width
        images = np.flip(images, 2).copy()
        binary_masks = np.flip(binary_masks, 3).copy()
        if ignore_masks is not None:
            ignore_masks = np.flip(ignore_masks, 2).copy()

    return images, binary_masks, ignore_masks


def apply_resize_scale(
        images: np.ndarray, 
        binary_masks: np.ndarray, 
        ignore_masks: np.ndarray|None,
        min_scale: float,
        max_scale: float,
        target_dims: Tuple[int],
):
    """
    Takes target size as input and randomly scales the given target size between `min_scale`
    and `max_scale`. It then scales the input image such that it fits inside the scaled target
    box, keeping the aspect ratio constant.

    For RGB image, it uses bilinear interpolation, and for binary masks, it uses nearest 
    neighbour interpolation.

    Args:
        images: [T, H, W, 3]
        binary_masks: [T, N, H, W]
        ignore_masks: [T, H, W]
        min_scale, max_scale: floats, range to pick a random scale to be applied
        target_dims: Tuple[int, int], target resolution
    """
    scale = np.random.uniform(min_scale, max_scale)
    
    input_size = images.shape[1:3]
    # new target size given a scale
    target_scale_size = np.multiply(target_dims, scale)
    # Compute actual rescaling applied to input image and output size
    output_scale = np.minimum(target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1])
    output_size = np.round(np.multiply(input_size, output_scale)).astype(int)

    pad_h = 0
    pad_w = 0
    if output_size[0] > target_dims[0]:
        output_size[0] = target_dims[0]
    
    if output_size[1] > target_dims[1]:
        output_size[1] = target_dims[1]
    
    if output_size[0] < target_dims[0]:
        pad_h = target_dims[0] - output_size[0]
    
    if output_size[1] < target_dims[1]:
        pad_w = target_dims[1] - output_size[1]

    # resize
    images = resize_images(images, output_size[0], output_size[1])
    binary_masks = resize_masks(binary_masks, output_size[0], output_size[1], binary=True)
    if ignore_masks is not None:
        ignore_masks = resize_masks(ignore_masks, output_size[0], output_size[1], binary=False)

    h,w = binary_masks.shape[2:]
    padding_mask = np.zeros((h,w), dtype=np.uint8)
    # pad
    if pad_h>0 or pad_w>0:
        im_pad = ((0,0), (0, pad_h), (0, pad_w), (0,0))
        images = np.pad(images, im_pad, mode='constant', constant_values=128.0)
        mask_pad = ((0,0), (0,0), (0, pad_h), (0, pad_w))
        binary_masks = np.pad(binary_masks, mask_pad, mode='constant', constant_values=0)
        padding_mask = np.pad(padding_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=1)
        if ignore_masks is not None:
            mask_pad = ((0,0), (0, pad_h), (0, pad_w))
            ignore_masks = np.pad(ignore_masks, mask_pad, mode='constant', constant_values=0)

    return images, binary_masks, ignore_masks, padding_mask


def mask_to_bbox(masks: Tensor, raise_error_if_null_mask: Optional[bool] = True) -> torch.Tensor:
    """
    Extracts bounding boxes from masks
    
    Args:
        masks: tensor of shape [N_1, ..., N_d, H, W]
        raise_error_if_null_mask: Flag for whether or not to raise an error if a mask is all-zeros.
    
    Returns: torch.Tensor of shape [N, 4] containing bounding boxes coordinates in [x, y, w, h] format.
            If `raise_error_if_null_mask` is False, coordinates [-1, -1, -1, -1] will be returned for all-zeros masks.
    """
    assert masks.ndim > 2

    # flatten additional leading dims
    leading_dim_sizes = masks.shape[:-2]
    masks = masks.reshape(-1, *masks.shape[-2:])  # [N, H, W]
    assert masks.ndim == 3  # sanity check

    null_masks = torch.logical_not(torch.any(masks.flatten(1), 1))[:, None]  # [N, 1]
    if torch.any(null_masks) and raise_error_if_null_mask:
        raise ValueError("One or more all-zero masks found")

    h, w = masks.shape[-2:]

    reduced_rows = torch.any(masks, 2).long()  # [N, H]
    reduced_cols = torch.any(masks, 1).long()  # [N, W]

    x_min = (reduced_cols * torch.arange(-w-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]
    y_min = (reduced_rows * torch.arange(-h-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]

    x_max = (reduced_cols * torch.arange(w, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]
    y_max = (reduced_rows * torch.arange(h, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    bbox_coords = torch.stack((x_min, y_min, width, height), 1)
    invalid_box = torch.full_like(bbox_coords, -1)

    bbox_coords = torch.where(null_masks, invalid_box, bbox_coords)  # [N, 4]
    return bbox_coords.reshape(*leading_dim_sizes, 4)  # [..., 4]


def mask_to_bbox(masks: Tensor, raise_error_if_null_mask: Optional[bool] = True) -> torch.Tensor:
    """
    Extracts bounding boxes from masks
    
    Args:
        masks: tensor of shape [N_1, ..., N_d, H, W]
        raise_error_if_null_mask: Flag for whether or not to raise an error if a mask is all-zeros.
    
    Returns: torch.Tensor of shape [N, 4] containing bounding boxes coordinates in [x, y, w, h] format.
            If `raise_error_if_null_mask` is False, coordinates [-1, -1, -1, -1] will be returned for all-zeros masks.
    """
    assert masks.ndim > 2

    # flatten additional leading dims
    leading_dim_sizes = masks.shape[:-2]
    masks = masks.reshape(-1, *masks.shape[-2:])  # [N, H, W]
    assert masks.ndim == 3  # sanity check

    null_masks = torch.logical_not(torch.any(masks.flatten(1), 1))[:, None]  # [N, 1]
    if torch.any(null_masks) and raise_error_if_null_mask:
        raise ValueError("One or more all-zero masks found")

    h, w = masks.shape[-2:]

    reduced_rows = torch.any(masks, 2).long()  # [N, H]
    reduced_cols = torch.any(masks, 1).long()  # [N, W]

    x_min = (reduced_cols * torch.arange(-w-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]
    y_min = (reduced_rows * torch.arange(-h-1, -1, dtype=torch.long, device=masks.device)[None]).argmin(1)  # [N]

    x_max = (reduced_cols * torch.arange(w, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]
    y_max = (reduced_rows * torch.arange(h, dtype=torch.long, device=masks.device)[None]).argmax(1)  # [N]

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    bbox_coords = torch.stack((x_min, y_min, width, height), 1)
    invalid_box = torch.full_like(bbox_coords, -1)

    bbox_coords = torch.where(null_masks, invalid_box, bbox_coords)  # [N, 4]
    return bbox_coords.reshape(*leading_dim_sizes, 4)  # [..., 4]


def apply_random_crop(
        images: np.ndarray,
        binary_masks: np.ndarray,
        ignore_masks: np.ndarray|None,
        crop_size: Tuple[int]
):
    """
    Apply random crops on the input tensors. Cropping must preserve all of the foreground 
    mask region. Mask is preserved by comparing desired crop dimensions with the smallest
    bounding box that covers all of the objects present in the first frame of the clip.

    If crop size is larger than the input, apply padding.

    If crop size is smaller than the input, one of the following two cases may apply:
    A. Crop is mask preserving, no further complication
    B. Crop cuts some non-zero mask region, i.e., bounding box is larger than the crop area.
       In that case, scale the crop size to match the bounding box and shrink the cropped 
       area to match original crop size.

    Args:
        images: [T, H, W, 3]
        binary_masks: [T, N, H, W]
        ignore_masks: [T, H, W]
        crop_size: randomly crop area of this dimension
    """

    assert images.ndim == 4 and binary_masks.ndim == 4
    crop_height, crop_width = crop_size

    # find bounding box of the first frame masks
    ref_mask = np.any(binary_masks[0], 0)
    ref_mask = torch.from_numpy(np.ascontiguousarray(ref_mask))
    # bbox corners
    x1, y1, box_w, box_h = mask_to_bbox(ref_mask.unsqueeze(0), raise_error_if_null_mask=True)[0].tolist()
    x2 = x1 + box_w
    y2 = y1 + box_h

    expanded_crop = False
    if box_w >= crop_width or box_h >= crop_height:
        # cropping cuts the mask, resize and crop
        crop_dilate = max(box_h/crop_height, box_w/crop_width)
        crop_height, crop_width = round(crop_size[0]*crop_dilate), round(crop_size[1]*crop_dilate)
        expanded_crop = True

    im_height, im_width = ref_mask.shape[-2:]
    padding_mask = np.zeros((im_height, im_width))
    
    # start offset for crop window
    x_min = max(0, x2 - crop_width)
    x_max = min(im_width - crop_width, x1)
    y_min = max(0, y2 - crop_height)
    y_max = min(im_height - crop_height, y1)

    if x_max < x_min or y_max < y_min:
        # crop size larger than input, so apply padding
        x_pad, y_pad = 0,0
        if x_max < x_min:
            crop_x1 = 0
            x_pad = crop_width - im_width
        else:
            crop_x1 = random.randint(x_min, x_max)
        if y_max < y_min:
            crop_y1 = 0
            y_pad = crop_height - im_height
        else:
            crop_y1 = random.randint(y_min, y_max)
        crop_x2, crop_y2 = crop_x1 + crop_width, crop_y1 + crop_height

        im_pad = ((0,0), (0, y_pad), (0, x_pad), (0,0))
        mask_pad = ((0,0), (0,0), (0, y_pad), (0, x_pad))
        images = np.pad(images, im_pad, mode='constant', constant_values=128.0)
        binary_masks = np.pad(binary_masks, mask_pad, mode='constant', constant_values=0)
        if ignore_masks is not None:
            mask_pad = ((0,0), (0, y_pad), (0, x_pad))
            ignore_masks = np.pad(ignore_masks, mask_pad, mode='constant', constant_values=0)
        padding_mask = np.pad(padding_mask, ((0, y_pad), (0, x_pad)), mode='constant', constant_values=1)

    else:
        crop_x1 = random.randint(x_min, x_max)
        crop_y1 = random.randint(y_min, y_max)
        crop_x2, crop_y2 = crop_x1 + crop_width, crop_y1 + crop_height
    
    # crop
    images = images[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
    binary_masks = binary_masks[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    if ignore_masks is not None:
        ignore_masks = ignore_masks[:, crop_y1:crop_y2, crop_x1:crop_x2]
    padding_mask = padding_mask[crop_y1:crop_y2, crop_x1:crop_x2]

    if expanded_crop:
        # resize the cropped tensors to orig crop size
        images = resize_images(images, crop_size[0], crop_size[1])
        binary_masks = resize_masks(binary_masks, crop_size[0], crop_size[1], binary=True)
        if ignore_masks is not None:
            ignore_masks = resize_masks(ignore_masks, crop_size[0], crop_size[1], binary=False)
        padding_mask = resize_masks(np.expand_dims(padding_mask, 0), crop_size[0], crop_size[1])[0]
    
    return images, binary_masks, ignore_masks, padding_mask
    

    