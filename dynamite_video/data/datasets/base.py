import os
import cv2
import math
import torch
import random
import itertools
import numpy as np
import pycocotools.mask as mt
import torch.nn.functional as F
import imgaug.augmenters as iaa

from PIL import Image
from einops import rearrange
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset, ConcatDataset as _ConcatDataset


from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.data.utils.data_utils import apply_resizer
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.data.utils.clicker import get_clicks_coords_evaluation


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
    

    def random_draw(self, x):
        while True:
            yield random.randint(0, x)
    

    def create_training_samples(
        self, 
        videos: Dict[str, GenericVideoSequence],
        num_total_samples: int,
        frame_sampling_multiplicative_factor: float,
        max_num_instances: int=4,
    ):
        """
        Generate a set number of samples from the dataset

        Args:
            videos: all video sequences in the dataset as GenericVideoSequence objects
            num_total_samples: total num of samples to be drawn from the dataset
            frame_sampling_multiplicative_factor: defines window to draw frames of a clip from
            max_num_instances: maximum #instances permitted in a clip, default: 4
        """

        # fix seed so that same set of samples is generated across all processes
        rnd_state_backup = random.getstate()
        random.seed(2202)

        max_temporal_span = int(round(frame_sampling_multiplicative_factor * self.clip_length))

        # uniformly sample videos with different instance counts
        samples_per_instance_count = [int(math.ceil(num_total_samples / float(max_num_instances))) for _ in range(max_num_instances)]
        leftovers = num_total_samples - sum(samples_per_instance_count)
        samples_per_instance_count[-1] += leftovers

        video_bins = defaultdict(list)
        for vid_id, vid in videos.items():
            bin_id = min(len(vid.instance_ids), max_num_instances)
            for b in range(1, bin_id+1):
                video_bins[b].append(vid_id)


        samples_by_num_instance = defaultdict(list)
        for bin_id in range(1, max_num_instances+1):
            # calculate how many samples with the given number of 
            # instances must be drawn from each available video
            num_available_videos = len(video_bins[bin_id])
            if num_available_videos==0:
                continue
            samples_per_video = int(math.ceil(samples_per_instance_count[bin_id-1] / num_available_videos))

            for vid_id, vid in videos.items():
                if vid_id not in video_bins[bin_id]:
                    continue
                
                # last index in the video that can be the first frame of a sampled clip
                last_t = len(vid) - self.clip_length
                index_generator = self.random_draw(last_t)
                count = 0

                while True:
                    t = next(index_generator)

                    if len(vid.segmentations[t].keys()) < bin_id:
                        continue

                    valid_instance_ids = [iid for iid in vid.instance_ids if iid in vid.segmentations[t]]
                    if not valid_instance_ids:
                        continue

                    samples_by_num_instance[bin_id].append((vid_id, t, valid_instance_ids))
                    count += 1
                    if count>=samples_per_video:
                        break


        # random samples within each bin
        for bin_id in samples_by_num_instance.keys():
            random.shuffle(samples_by_num_instance[bin_id])

        train_samples = []
        train_sample_dims = []
        train_sample_bin = []

        for bin_id in range(1, max_num_instances+1):

            for sample in samples_by_num_instance[bin_id]:

                # extract metadata for a training sample (i.e., a clip)
                # 1. the video ID to identify the source video to extract the clip from
                # 2. index of the frame in the video that will be the first frame of the clip - 
                # let's call it a reference frame
                # 3. IDs of the instances in this reference frame
                vid_id, ref_frame_idx, instance_ids = sample

                assert len(instance_ids) >= bin_id
                if len(instance_ids) > bin_id:
                    # if reference frame contains more instances than permitted, take a subset
                    # only happens when sampling from the final bin
                    instance_ids = random.sample(instance_ids, bin_id)

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
                train_sample_bin.append(bin_id)

        # restore initial random state
        random.setstate(rnd_state_backup)

        train_samples = train_samples[:num_total_samples]
        train_sample_dims = train_sample_dims[:num_total_samples]
        train_sample_bin = train_sample_bin[:num_total_samples]

        return train_samples, train_sample_dims, train_sample_bin


        # samples_by_num_instance = defaultdict(list)
        # for vid_id, vid in videos.items():
        #     # last index in the video to be the first frame of a clip
        #     last_t = len(vid) - self.clip_length

        #     for t in range(last_t):
        #         # instances present in the clip starting at frame t,
        #         # are the instances present in frame t
        #         valid_instance_ids = [iid for iid in vid.instance_ids if iid in vid.segmentations[t]]
        #         if not valid_instance_ids:
        #             continue
                
        #         # max_num_instances is currently set to 4, TarViS default
        #         bin_id = min(max_num_instances, len(valid_instance_ids))

        #         # n-th bin has clips with n instances, except the final bin (say bin # N)
        #         # final bin has clips with N or more instances
        #         samples_by_num_instance[bin_id].append((vid_id, t, valid_instance_ids))

        # # random samples within each bin
        # for bin_id in samples_by_num_instance.keys():
        #     random.shuffle(samples_by_num_instance[bin_id])
        
        # train_samples = []
        # train_sample_dims = []
        # train_sample_ni = []
        
        # # uniformly sample videos with different instance counts
        # num_instances_per_count = int(math.ceil(num_total_samples / float(max_num_instances)))
        # available_sample_pool = []

        # for ni in range(max_num_instances, 0, -1):
        #     if ni not in samples_by_num_instance.keys():
        #         continue
        #     available_sample_pool = samples_by_num_instance[ni] + available_sample_pool

        #     # TODO - exclude extracted samples from the pool
        #     for ii in range(num_instances_per_count):
        #         ii = ii % len(available_sample_pool)
                
        #         # extract metadata for a training sample (i.e., a clip)
        #         # 1. the video ID to identify the source video to extract the clip from
        #         # 2. index of the frame in the video that will be the first frame of the clip - 
        #         # let's call it a reference frame
        #         # 3. IDs of the instances in this reference frame
        #         vid_id, ref_frame_idx, instance_ids = available_sample_pool[ii]

        #         assert len(instance_ids) >= ni

        #         if len(instance_ids) > ni:
        #             # if reference frame contains more instances than permitted, take a subset
        #             # only happens when sampling from the final bin
        #             instance_ids = random.sample(instance_ids, ni)

        #         # extract the video info from the dataset
        #         vid = videos[vid_id]

        #         # the rest of the frames in the clip can be sampled from 
        #         # a window of the next max_temporal_span frames
        #         other_frames_list = list(
        #             range(ref_frame_idx + 1, min(len(vid), ref_frame_idx + max_temporal_span))
        #         )

        #         assert len(other_frames_list) >= self.clip_length - 1, \
        #             f"Something went wrong here: {ref_frame_idx}, {len(vid)}, {other_frames_list}"

        #         # from the window sample the necessary #frames to fill the clip
        #         other_frame_idxes = sorted(random.sample(other_frames_list, self.clip_length - 1))

        #         train_samples.append({
        #             "vid_id": vid_id,
        #             "ref_frame": ref_frame_idx,
        #             "other_frames": other_frame_idxes,
        #             "ref_inst_ids": instance_ids,
        #             "video": vid,
        #         })

        #         # store original clip resolution
        #         train_sample_dims.append((vid.height, vid.width))
        #         # store bin (==#instances in the clip) size
        #         train_sample_ni.append(ni)

        # # restore initial random state
        # random.setstate(rnd_state_backup)

        # train_samples = train_samples[:num_total_samples]
        # train_sample_dims = train_sample_dims[:num_total_samples]
        # train_sample_ni = train_sample_ni[:num_total_samples]

        # return train_samples, train_sample_dims, train_sample_ni

    
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
    def __init__(
        self, 
        cfg, 
        name: str, 
        clip_length:int, 
        fps:int, 
        num_overlapping_frames: int,
        split: str
    ):
        """
        Initialize with dataset metadata

        Args:
            cfg: configuration
            name: name of the dataset
            clip_length: length of each training sample from the dataset
            fps: video fps
            num_overlapping_frames: number of overlapping frames between successive clips
            split: distinguish "val" or "test"
        """
        super().__init__()

        assert clip_length >= 1

        self.cfg = cfg
        self.name = name
        self.clip_length = clip_length
        self.fps = fps
        self.num_overlapping_frames = num_overlapping_frames
        self.split = split

        self.sample_image_dims = []
        self.sample_instance_counts = []

    def create_inference_clips(self, annotations):
        
        all_clips = []
        
        for seq in annotations["sequences"]:
            
            # clip indices
            seq_length = len(seq["image_paths"])
            
            indices = []
            step = self.clip_length - self.num_overlapping_frames
            start = 0
            while start + self.clip_length <= seq_length:
                indices.append(tuple(range(start, start + self.clip_length)))
                start += step

            if indices[-1][-1] != seq_length - 1:
                indices.append(tuple(range(indices[-1][-1] - self.num_overlapping_frames +1, seq_length)))
            
            seq["clip_indices"] = indices

            # dimensions of the frames in the sequence - for RLE decoding
            img_dims = seq["height"], seq["width"]

            # IDs of the instances present in the sequence
            seq_instances = list(seq["categories"].keys())

            # serialize instance IDs
            self.orig_to_serial_ids, self.serial_to_orig_ids = self.serialize_instance_ids(seq_instances)

            sequence_clips = []
            click_start_t = 1
            for clip_indices in indices:
                
                clip_images = []
                clip_masks = []
                
                # load images
                clip_images = [np.asarray(Image.open(seq["image_paths"][t]).convert("P")) for t in clip_indices]
                clip_images = np.stack(clip_images)     # [T,H,W]

                # load masks
                clip_rles = [seq["segmentations"][t] for t in clip_indices]
                clip_masks, clip_instances, frame_instance_occupancy = self.prepare_masks(clip_rles, img_dims, seq_instances)
                clip_masks = np.stack([np.stack(masks_t) for masks_t in clip_masks]).astype('uint8')    # [T,N,H,W]
                
                # apply resizing
                clip_images, clip_masks = apply_resizer(clip_images,
                                                        clip_masks,
                                                        mode='min_dim',
                                                        min_dim=self.cfg.INPUT.AUGMENTATION.MIN_DIM_TEST,
                                                        max_dim=self.cfg.INPUT.AUGMENTATION.MAX_DIM_TEST,
                                                    )
                # background masks
                bg_masks = []
                for fr_masks in clip_masks:
                    dummy = np.ones_like(fr_masks[0])
                    for inst_mask in fr_masks:
                        dummy[np.where(inst_mask)==1] = 0
                    bg_masks.append(dummy)
                bg_masks = np.stack(bg_masks).astype('uint8')   # [T,H,W]

                # get clicks
                num_clicks_per_object, fg_coords_list, bg_coords_list, max_timestamp = get_clicks_coords_evaluation(
                                                                                                instance_masks=clip_masks,
                                                                                                clip_instance_ids=clip_instances,
                                                                                                sequence_instance_ids=list(self.serial_to_orig_ids.keys()),
                                                                                                frame_instance_occupancy=frame_instance_occupancy,
                                                                                                max_num_points=1,
                                                                                                first_click_center=True,
                                                                                                start_t=click_start_t,
                                                                                            )
                
                click_start_t = max(max_timestamp) + 1

                num_instances_per_frame = [len(fr_inst) for fr_inst in clip_instances]

                entry = {
                    "indices": clip_indices,
                    "images": clip_images,
                    "num_instances_per_frame": num_instances_per_frame,
                    "num_clicks_per_object": num_clicks_per_object,
                    "fg_coords_list": fg_coords_list,
                    "bg_coords_list": bg_coords_list,
                    "max_timestamp_list": max_timestamp,
                    "instance_masks": clip_masks,
                    "bg_masks": bg_masks,
                    "padding_mask": np.zeros((clip_images.shape[1], clip_images.shape[2])).astype('uint8')
                }
                sequence_clips.append(entry)
            
            all_clips.append(sequence_clips)
        
        return all_clips
    

    def prepare_masks(self, clip_rles, size, instance_ids):
        
        clip_masks = []
        clip_instances = []
        frame_instance_occupancy = defaultdict(list)

        for fr_idx, frame_rles in enumerate(clip_rles):
            # store binary instance masks of each frame
            frame_masks = []
            frame_instances = []
            
            for inst_id in instance_ids:
                # check for binary mask of each instance of the sequence
                # pad empty for ones that are absent

                if inst_id in frame_rles.keys():
                    frame_masks.append(self.decode_mask(frame_rles[inst_id], size))
                    frame_instances.append(self.orig_to_serial_ids[inst_id])
                    frame_instance_occupancy[self.orig_to_serial_ids[inst_id]].append(fr_idx)
                else:
                    frame_masks.append(np.zeros(size))
            
            clip_masks.append(frame_masks)
            clip_instances.append(frame_instances)
        return clip_masks, clip_instances, frame_instance_occupancy
            

    def serialize_instance_ids(self, orig_ids):
        """
        Serialize instance IDs. IDs are 1-indexed to avoid conflict in semantic mask
        with background pixels (0)

        Args:
            orig_ids: original instance IDs, potentially non-sequential

        Returns:
            orig_to_serial_id: mapping from original IDs to sequential IDs
            serial_to_orig_id: mapping from sequential IDs to original IDs
        """
        orig_ids = sorted(orig_ids)
        serial_ids = [i for i in range(1, len(orig_ids)+1)]
        serial_to_orig_id = OrderedDict(zip(serial_ids, orig_ids))
        orig_to_serial_id = OrderedDict(zip(orig_ids, serial_ids))
        return orig_to_serial_id, serial_to_orig_id
    
    
    def decode_mask(self, encoded_mask: Union[str, List[int]], size=None):
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



class ConcatDataset(_ConcatDataset):
    """
    A dataset class to concatenate multiple sub-datasets into a single dataset
    """
    def __init__(self, datasets: List[TrainingDataset]):
        super().__init__(datasets)

        self.sample_image_dims = list(itertools.chain(*[ds.sample_image_dims for ds in self.datasets]))

    def sample_dataset_names(self):
        return list(itertools.chain(*[ds.sample_dataset_names() for ds in self.datasets]))