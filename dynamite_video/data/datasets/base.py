import os
import cv2
import math
import torch
import random
import itertools
import numpy as np
import torch.nn.functional as F
import imgaug.augmenters as iaa

from einops import rearrange
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from collections import defaultdict
from torch.utils.data import Dataset, ConcatDataset as _ConcatDataset


from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks


class TrainingDataset(Dataset, ABC):
    """
    Base Training Dataset Class
    """
    
    def __init__(
        self, 
        cfg, 
        name: str, 
        clip_length:int, 
        num_samples: int, 
        fps:int, 
        frame_sampling_multiplicative_factor: float
    ):
        """
        Initialize with dataset metadata

        Args:
            cfg: configuration
            name: name of the dataset
            clip_length: length of each training sample from the dataset
            num_samples: num of training samples drawn from the dataset
            fps: video fps
            frame_sampling_multiplicative_factor: defines the window from 
                where clip frames are sampled relative to first frame
        """
        super().__init__()

        assert clip_length >= 1
        assert num_samples >= 1
        assert frame_sampling_multiplicative_factor >= 1

        self.cfg = cfg
        self.name = name
        self.clip_length = clip_length
        self.num_samples = num_samples
        self.fps = fps
        self.frame_sampling_multiplicative_factor = frame_sampling_multiplicative_factor

        self.sample_image_dims = []
        self.sample_instance_counts = []
    

    def __len__(self):
        return len(self.sample_image_dims)
    
    def __getitem__(self, index):
        n_tries = 0
        while True:

            sample = self.parse_sample(index)
            if sample is not None:
                return sample

            num_instances = self.sample_instance_counts[index]
            self.fallback_candidates[num_instances].discard(index)
            index = random.choice(list(self.fallback_candidates[num_instances]))
            n_tries += 1
            if n_tries % 3 == 0:
                print(f"Num failed tries = {n_tries} for dataset {self.name}, num_instances {num_instances}")
    

    def parse_sample(self, index):
        """
        Prepare a sample (training clip) for training forward pass

        Args:
            index: index of the sample to return. The index addresses a sample, 
                which is a dict with the following keys:
                `vid_id`: id of the source video
                `ref_frame`: first frame of the clip
                `other_frames`: list of rest of the frames in the clip
                `ref_inst_ids`: instances present in the clip

        """
        sample = self.samples[index]
        video = self.videos[sample['vid_id']]

        # get absolute frame indices in the video that will form a clip
        frame_indices = [sample["ref_frame"]] + sample["other_frames"]
        clip_inst_ids = sample["ref_inst_ids"]
        clip_id = sample["vid_id"]
        
        # extract clip from the video
        clip = video.extract_subsequence(frame_indices, clip_inst_ids, clip_id)

        # load RGB frames as a list of np.ndarrays, each [H, W, 3]
        images = clip.load_images()
        
        # load binary and semantic masks as lists
        masks, semantic_masks, instance_maps = clip.prepare_masks()

        # color augmentations
        images = self.apply_color_augmentation(images)

        images = np.stack(images)                                       # [T, H, W, 3]
        masks = np.stack([np.stack(masks_t) for masks_t in masks])      # [T, N, H, W]
        masks = np.transpose(masks, (1, 0, 2, 3))                       # [N, T, H, W]
        semantic_masks = np.stack(semantic_masks)                       # [T, H, W]

        meta_info = {
            "orig_dims": images[0].shape[:2],
            "seq_name": video.id,
            "frame_indices": frame_indices,
            "instance_maps": instance_maps,
        }

        # data augmentations
        # resize shortest edge
        images, masks, semantic_masks = self.resize_shortest_edge(images, masks, semantic_masks)
        # apply horizontal flip
        images, masks, semantic_masks = self.apply_random_horizontal_flip(images, masks, semantic_masks)
        # apply random crops
        images, masks, semantic_masks, padding_mask = self.apply_random_crop(images, masks, semantic_masks)
        bg_masks = np.logical_not(np.logical_or(padding_mask, semantic_masks)).astype('uint8')

        images = np.transpose(images, (0, 3, 1, 2))                  # [T, H, W, 3] -> [T, 3, H, W]
        if self.cfg.INPUT.RGB and self.cfg.INPUT.AUGMENTATION.RGB_TO_BGR:
            images = np.flip(images, 1).copy()                       # RGB -> BGR

        return {
            "images": images,
            "instance_masks": masks,
            "semantic_masks": semantic_masks,
            "padding_mask": padding_mask,
            "bg_masks": bg_masks,
            "ref_frame_index": 0,
            "dataset": self.name,
            "meta": meta_info
        }

    
    @abstractmethod
    def map_annotations(self, *args, **kwargs):
        """Map dataset-specific annotation info to a common dataset dictionary format"""
        pass


    def sample_dataset_names(self):
        return [self.name] * len(self.sample_image_dims)
    

    def create_training_samples(
        self, 
        videos: Dict[str, GenericVideoSequence],
        num_total_samples: int,
        frame_sampling_multiplicative_factor: float,
        max_num_instances: int=4,
    ):
        """
        Generate set number of samples from the dataset

        Args:
            videos: all video sequences in the dataset as GenericVideoSequence objects
            num_total_samples: total num of samples to be drawn from the dataset
            frame_sampling_multiplicative_factor: defines window to draw frames of a clip from
            max_num_instances: maximum #instances permitted in a clip, default: 4
        """

        # fix seed so that same set of samples is generated across all processes
        rnd_state_backup = random.getstate()
        random.seed(2202)

        samples_by_num_instance = defaultdict(list)
        max_temporal_span = int(round(frame_sampling_multiplicative_factor * self.clip_length))

        for vid_id, vid in videos.items():
            # last index in the video to be the first frame of a clip
            last_t = len(vid) - self.clip_length

            for t in range(last_t):
                # instances present in the clip starting at frame t,
                # are the instances present in frame t
                valid_instance_ids = [iid for iid in vid.instance_ids if iid in vid.segmentations[t]]
                if not valid_instance_ids:
                    continue
                
                # max_num_instances is currently set to 4, TarViS default
                bin_id = min(max_num_instances, len(valid_instance_ids))

                # n-th bin has clips with n instances, except the final bin (say bin # N)
                # final bin has clips with N or more instances
                samples_by_num_instance[bin_id].append((vid_id, t, valid_instance_ids))

        # random samples within each bin
        for bin_id in samples_by_num_instance.keys():
            random.shuffle(samples_by_num_instance[bin_id])
        
        train_samples = []
        train_sample_dims = []
        train_sample_ni = []
        
        # uniformly sample videos with different instance counts
        num_instances_per_count = int(math.ceil(num_total_samples / float(max_num_instances)))
        available_sample_pool = []

        for ni in range(max_num_instances, 0, -1):
            if ni not in samples_by_num_instance.keys():
                continue
            available_sample_pool = samples_by_num_instance[ni] + available_sample_pool

            # TODO - exclude extracted samples from the pool
            for ii in range(num_instances_per_count):
                ii = ii % len(available_sample_pool)
                
                # extract metadata for a training sample (i.e., a clip)
                # 1. the video ID to identify the source video to extract the clip from
                # 2. index of the frame in the video that will be the first frame of the clip - 
                # let's call it a reference frame
                # 3. IDs of the instances in this reference frame
                vid_id, ref_frame_idx, instance_ids = available_sample_pool[ii]

                assert len(instance_ids) >= ni

                if len(instance_ids) > ni:
                    # if reference frame contains more instances than permitted, take a subset
                    # only happens when sampling from the final bin
                    instance_ids = random.sample(instance_ids, ni)

                # extract the video info from the dataset
                vid = videos[vid_id]

                # the rest of the frames in the clip can be sampled from 
                # a window of the next max_temporal_span frames
                other_frames_list = list(
                    range(ref_frame_idx + 1, min(len(vid), ref_frame_idx + max_temporal_span))
                )

                assert len(other_frames_list) >= self.clip_length - 1, \
                    f"Something went wrong here: {ref_frame_idx}, {len(vid)}, {other_frames_list}"

                # from the window sample the necessary #frames to fill the clip
                other_frame_idxes = sorted(random.sample(other_frames_list, self.clip_length - 1))

                train_samples.append({
                    "vid_id": vid_id,
                    "ref_frame": ref_frame_idx,
                    "other_frames": other_frame_idxes,
                    "ref_inst_ids": instance_ids,
                    "video": vid,
                })

                # store original clip resolution
                train_sample_dims.append((vid.height, vid.width))
                # store bin (==#instances in the clip) size
                train_sample_ni.append(ni)

        # restore initial random state
        random.setstate(rnd_state_backup)

        train_samples = train_samples[:num_total_samples]
        train_sample_dims = train_sample_dims[:num_total_samples]
        train_sample_ni = train_sample_ni[:num_total_samples]

        return train_samples, train_sample_dims, train_sample_ni

    
    def apply_color_augmentation(self, images: List[np.ndarray]):
        """
        Apply same color augmentation to all frames

        Args:
            images: list of RGB images [H, W, 3]
        """
        # if color augmentation is disabled
        if not self.cfg.INPUT.AUGMENTATION.COLOR_AUG:
            return images
        
        color_augmenter = iaa.Sequential([
            iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            iaa.AddToBrightness(add=(-25, 25))
        ])
        det_augmenter = color_augmenter.to_deterministic()
        return [det_augmenter(image=img) for img in images]

    
    def resize_shortest_edge(
        self, 
        images: np.ndarray, 
        binary_masks: np.ndarray,
        semantic_masks: np.ndarray
    ):
        """
        Resize video frames to a specified resolution .Shortest edge of each frame 
        is reduced to the target size.

        Args:
            images: [T, H, W, 3]
            binary_masks: [N, T, H, W]
            semantic_masks: [T, H, W]
        """
        mode = self.cfg.INPUT.AUGMENTATION.RESIZE_TRAIN
        ALLOWED_MODES = ["min_dim"]
        assert mode in ALLOWED_MODES, f"Desired resize mode {mode} is not available. \
            Choose from {ALLOWED_MODES}"

        if mode == "none":
            return images, binary_masks, semantic_masks
        
        # params for min dim mode (resize shortest edge)
        max_dim = self.cfg.INPUT.AUGMENTATION.MAX_DIM_TRAIN
        min_dim = self.cfg.INPUT.AUGMENTATION.MIN_DIM_TRAIN

        # compute target resolution
        new_height, new_width = compute_resized_dims(
            *images.shape[1:3], 
            min_dim, 
            max_dim
        )

        images = resize_images(images, new_height, new_width)

        semantic_masks = resize_masks(semantic_masks, new_height, new_width, binary=False)

        binary_masks = resize_masks(binary_masks, new_height, new_width, binary=True)

        return images, binary_masks, semantic_masks
    
    
    def apply_random_horizontal_flip(
        self, 
        images: np.ndarray, 
        binary_masks: np.ndarray,
        semantic_masks: np.ndarray
    ):
        """
        Apply random horizontal flips
        
        Args:
            images: [T, H, W, 3]
            binary_masks: [N, T, H, W]
            semantic_masks: [T, H, W]
        """
        assert images.ndim == 4 and binary_masks.ndim == 4 and semantic_masks.ndim == 3

        # if random flips are disabled
        if not self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP:
            return images, binary_masks, semantic_masks
        
        # only horizontal flips
        assert self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_AXIS == "horizontal", f"Only 'horizontal' flips are allowed, {self.cfg.INPUT.RANDOM_FLIP_AXIS} is not allowed!"
        
        # flip probability
        prob = self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_PROB

        if torch.rand(1) < prob:
            # flip along width
            images = np.flip(images, 2).copy()
            binary_masks = np.flip(binary_masks, 3).copy()
            semantic_masks = np.flip(semantic_masks, 2).copy()

        return images, binary_masks, semantic_masks


    def apply_random_crop(
        self, 
        images: np.ndarray, 
        binary_masks: np.ndarray, 
        semantic_masks: np.ndarray, 
    ):
        """
        Apply random horizontal flips
        
        Args:
            images: [T, H, W, 3]
            binary_masks: [N, T, H, W]
            semantic_masks: [T, H, W]
        """
        assert images.ndim == 4 and binary_masks.ndim == 4 and semantic_masks.ndim == 3

        # if random crops are disabled
        if not self.cfg.INPUT.AUGMENTATION.RANDOM_CROP:
            return images, binary_masks, semantic_masks
        
        # crop size 
        crop_size = (self.cfg.INPUT.AUGMENTATION.CROP_HEIGHT, self.cfg.INPUT.AUGMENTATION.CROP_WIDTH)

        # crop offsets
        input_size = images.shape[1:3]
        max_offset = np.subtract(input_size, crop_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)
        
        cropped_images = images[..., offset[0]:offset[0]+crop_size[0], offset[1]:offset[1]+crop_size[1], :]
        cropped_binary_masks = binary_masks[:, :, offset[0]:offset[0]+crop_size[0], offset[1]:offset[1]+crop_size[1]]
        cropped_semantic_masks = semantic_masks[:, offset[0]:offset[0]+crop_size[0], offset[1]:offset[1]+crop_size[1]]

        pad_size = np.subtract(crop_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        # account for applied mask
        padding_mask = np.ones(cropped_semantic_masks.shape[1:])
        if pad_size.sum() > 0:
            # image
            im_padding = ((0,0), (0, pad_size[0]), (0, pad_size[1]), (0,0))
            cropped_images = np.pad(cropped_images, im_padding, mode='constant', constant_values=128.0)

            # binary masks
            binary_mask_padding = ((0,0), (0,0), (0, pad_size[0]), (0, pad_size[1]))
            cropped_binary_masks = np.pad(cropped_binary_masks, binary_mask_padding, mode='constant', constant_values=0)

            # semantic masks
            semantic_mask_padding = ((0,0), (0, pad_size[0]), (0, pad_size[1]))
            cropped_semantic_masks = np.pad(cropped_semantic_masks, semantic_mask_padding, mode='constant', constant_values=0)

            padding = ((0, pad_size[0]), (0, pad_size[1]))
            padding_mask = np.pad(padding_mask, padding, mode='constant', constant_values=0)

        padding_mask = np.logical_not(padding_mask)
        return cropped_images, cropped_binary_masks, cropped_semantic_masks, padding_mask



class InferenceDataset(Dataset):
    ...




class ConcatDataset(_ConcatDataset):
    """
    A dataset class to concatenate multiple sub-datasets into a single dataset
    """
    def __init__(self, datasets: List[TrainingDataset]):
        super().__init__(datasets)

        self.sample_image_dims = list(itertools.chain(*[ds.sample_image_dims for ds in self.datasets]))

    def sample_dataset_names(self):
        return list(itertools.chain(*[ds.sample_dataset_names() for ds in self.datasets]))