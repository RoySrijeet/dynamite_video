import cv2
import itertools
import math
import random
import torch

import numpy as np

from abc import ABC
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset as _ConcatDatasetBase
from typing import Any, Dict, List

from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.data.utils.clicker import get_clicks_coords
from dynamite_video.data.utils.data_utils import (
    apply_color_augmentation, 
    apply_random_flip,
    apply_resize_scale,
)


######################### TRAINING DATASET BASE ###################################

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
        self.sample_object_counts = []
        self.MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        self.MAX_NUM_INSTANCES = self.cfg.TRAINING.MAX_NUM_INSTANCES

        self.fpack_reader = None
    

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        index = index % len(self.samples)
        n_tries = 0
        while True:
            try:
                sample = self.parse_sample(self.samples[index])
                return sample

            except:
                self.fallback_candidates.discard(index)
                index = random.choice(list(self.fallback_candidates))
                n_tries += 1
                if n_tries % 3 == 0:
                    print(f"Num failed tries = {n_tries} for dataset {self.name}")

    
    def parse_sample(self, sample):
        """
        Prepare a sample (training clip) for training forward pass.

        Args:
            sample: a dictionary with the following keys:
                `vid_id`: name of the source video sequence
                `ref_frame`: first ferame of the clip
                `other_frames`: list of rest of the frames in the clip
                `video`: source video cast as GenericVideoSequence object

        """
        # source video, a `GenericVideoSequence`
        video = sample['video']

        # get absolute frame indices in the video that will form a clip
        frame_indices = [sample["ref_frame"]] + sample["other_frames"]

        # extract clip from the video as a `GenericVideoSequence`
        clip = video.extract_subsequence(frame_indices, new_id=sample["vid_id"])

        # load images [T,H,W,3]
        images = clip.load_images()
        
        # load binary instance masks [T,N,H,W]; NOTE: ignore masks are not part of the binary masks
        binary_masks, ignore_masks = clip.prepare_masks()
        
        # color augmentations
        if self.cfg.INPUT.AUGMENTATION.COLOR_AUG:
            images = apply_color_augmentation(images)

        # data augmentations
        images, binary_masks, ignore_masks = apply_random_flip(images, 
                                                                binary_masks,
                                                                ignore_masks,
                                                                axis=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_AXIS,
                                                                prob=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_PROB,
                                                            )
        
        images, binary_masks, ignore_masks, padding_mask = apply_resize_scale(images,
                                                                binary_masks,
                                                                ignore_masks,
                                                                min_scale=self.cfg.INPUT.AUGMENTATION.MIN_SCALE,
                                                                max_scale=self.cfg.INPUT.AUGMENTATION.MAX_SCALE,
                                                                target_dims=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE,
                                                            )
        
        images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
        if self.cfg.INPUT.RGB:
            # BGR -> RGB
            images = np.flip(images, 1).copy()

        # remove small objects
        areas = binary_masks.sum(axis=(2,3))
        keep = (areas >= self.MIN_MASK_AREA).any(axis=0)
        assert keep.any(), f"All instances are below MIN_MASK_AREA (={self.MIN_MASK_AREA}) across all frames"
        binary_masks = binary_masks[:, keep, :, :]

        # select upto a set number of maximum objects
        if binary_masks.shape[1] > self.MAX_NUM_INSTANCES:
            keep = random.sample(range(binary_masks.shape[1]), self.MAX_NUM_INSTANCES)
            keep.sort()
            binary_masks = binary_masks[:, keep, :, :]
        
        # foreground masks
        T, _, H, W = binary_masks.shape
        panoptic_masks = np.zeros((T,H,W), dtype=np.uint8)      # T,H,W
        for fr_idx in range(T):
            for inst_id, inst_mask in enumerate(binary_masks[fr_idx]):
                panoptic_masks[fr_idx][inst_mask==1] = inst_id+1
        
        # NOTE: at this point, 0-labelled regions in foreground masks correspond
        # to the ignore mask area, the padding area, and any potential background
        
        # consider all parts of the image without fg label as bg
        bg_masks = (panoptic_masks==0) & ~padding_mask          # T,H,W

        if ignore_masks is None:
            ignore_masks = np.zeros_like(panoptic_masks)
        
        # sample clicks
        (num_clicks_per_target, 
         fg_coords_list, 
         bg_coords_list, 
         max_timestamp_list) = get_clicks_coords(target_masks=binary_masks, 
                                                max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                bg_masks=bg_masks,
                                                bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                gamma=self.cfg.CLICKER.TRAINING.GAMMA,
                                                start_t=1,
                                            )
        if not all(np.sum(num_clicks_per_target, axis=0)):
            raise "One or more targets in this clip did not receive a click!"

        meta_info = {
            "orig_dims": video.image_dims,
            "seq_name": f"{self.name}_{video.id}",
            "frame_indices": frame_indices,
            "orig_to_serial_id": clip.orig_to_serial_id, 
            "serial_to_orig_id": clip.serial_to_orig_id, 
            "ignore_class": clip.ignore_class,
            "target_categories": clip.object_categories,
            "max_instances_per_category": clip.meta_info.get('max_instances_per_category', None),
        }
        
        return {
            # T,3,H,W image tensors, not normalized, padded region has value 128
            "images": torch.as_tensor(images, dtype=torch.uint8),
            
            # T,N,H,W binary masks of target objects. This includes the semantic maps of the `stuff` classes 
            # and the instance maps of `thing` class instances. Which means, the binary masks do not include 
            # region of `thing` class that is not covered by corresponding instances, and the VOID class.
            "binary_masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
            
            # H,W boolean padding mask where padded region is labeled True
            "padding_mask": torch.as_tensor(padding_mask, dtype=torch.bool),
            
            # T,H,W boolean mask for `VOID` class
            "ignore_masks": torch.as_tensor(ignore_masks, dtype=torch.bool),
            
            # T,H,W bg mask includes any region not covered by the target object binary masks
            # NOTE: this includes ignore mask regions as well
            "bg_masks": torch.as_tensor(bg_masks, dtype=torch.uint8),
            
            # T,H,W semantic map with serialized target IDs. Background gets labeled 0
            "panoptic_masks": torch.as_tensor(panoptic_masks, dtype=torch.uint8),

            # T,N array recording num clicks on each target in each frame
            "num_clicks_per_object": num_clicks_per_target,
            
            # list of fg clicks sampled on the clip. Each click follows the format: [y,x,i,f,t]
            "fg_coords_list": fg_coords_list,
            
            # list of bg clicks sampled on the clip. Each click follows the format: [y,x,-1,f,t]
            "bg_coords_list": bg_coords_list,

            # list of length T recording timestamp of the latest click on each frame
            "max_timestamp_list": max_timestamp_list,

            # info about the clip and its source video
            "meta": meta_info,
        }
    
    
    def create_training_samples(
        self, 
        videos: Dict[str, GenericVideoSequence],
        num_total_samples: int,
    ):
        """
        Generate a set number of samples from the dataset

        Args:
            videos: all video sequences in the dataset as GenericVideoSequence objects
            num_total_samples: total num of samples to be drawn from the dataset
        """

        # fix seed so that same set of samples is generated across all processes
        rnd_state_backup = random.getstate()
        random.seed(2202)

        # given the starting frame index of a clip, the rest of the frames of the clip are randomly sampled from 
        # a temporal window of a certain size, specified by the configurable `frame_sampling_multiplicative_factor`
        max_temporal_span = int(round(self.frame_sampling_multiplicative_factor * self.clip_length))

        train_samples = []
        train_sample_dims = []
        
        sample_idx = 0
        num_samples_per_video = int(math.ceil(num_total_samples / len(videos)))
        for vid_id, vid in videos.items():
            
            # last index in the video to be the first frame of a clip
            last_t = len(vid) - self.clip_length

            for i in range(num_samples_per_video):
            
                # randomly select the first frame of the clip
                t = random.randint(0, last_t)
                
                # the rest of the frames in the clip can be sampled from a window of the next max_temporal_span frames
                other_frames_list = list(
                    range(t + 1, min(len(vid), t + max_temporal_span))
                )

                assert len(other_frames_list) >= self.clip_length - 1, f"Something went wrong here: {t}, {len(vid)}, {other_frames_list}"

                # from the window sample the necessary #frames to fill the clip
                other_frame_idxes = sorted(random.sample(other_frames_list, self.clip_length - 1))

                train_samples.append({
                    "idx": sample_idx,
                    "vid_id": vid_id,
                    "ref_frame": t,
                    "other_frames": other_frame_idxes,
                    "video": vid,
                })
                sample_idx += 1

                # store original clip resolution
                train_sample_dims.append((vid.height, vid.width))

        # restore initial random state
        random.setstate(rnd_state_backup)
        random.shuffle(train_samples)

        train_samples = train_samples[:num_total_samples]

        return train_samples


######################### EVALUATION DATASET BASE ###################################


class EvaluationDataset(Dataset):
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
        self.sample_object_counts = []

    
    def load_images(self, filepaths):
        """
        Given a list of filepaths, load JPEG images from the disc

        NOTE: the images are loaded in the default `cv2.imread(f, flags=cv2.IMREAD_COLOR)` 
        mode. This loads the image in BGR format, not RGB.

        Args:
            filepaths: list of paths to JPEG files
        
        Returns:
            np.ndarray of shape [T,H,W,3]
        """
        images = []
        for fp in filepaths:
            im = cv2.imread(fp, cv2.IMREAD_COLOR)
            if im is None:
                raise ValueError("No image found at path: {}".format(fp))
            images.append(im)
        
        images = np.stack(images)     # [T, H, W, 3]
        return images

    
    def load_png_masks(self, filepaths):
        """
        Given a list of filepaths, load PNG masks from the disc.

        Args:
            filepaths: list of paths to PNG files
        
        Returns:
            
        """
        # store semantic maps - expected shape [T,H,W]
        semantic_maps = []
        # store bg masks - expected shape [T,H,W]
        bg_masks = []
        # store object IDs present in each frame
        objects_per_frame = []
        # for each object, store index of the frame where it first appeared
        object_discovery = {}
        
        for fr_idx, fp in enumerate(filepaths):
            try:
                msk = np.asarray(Image.open(fp)).astype('uint8')
            except:
                raise RuntimeError(f"Problem loading mask from {fp}")
            
            semantic_maps.append(msk)
            
            # objects present in this frame
            objects = list(np.unique(msk))[1:]
            objects_per_frame.append(objects)

            # object discovery
            for obj_id in objects:
                if obj_id not in object_discovery.keys():
                    object_discovery[obj_id] = fr_idx
            
            # bg mask
            bg_msk = (msk == 0).astype('uint8')
            bg_masks.append(bg_msk)
        
        semantic_maps = np.stack(semantic_maps)
        bg_masks = np.stack(bg_masks)

        return semantic_maps, bg_masks, objects_per_frame, object_discovery
    

    def load_rle_masks(self):
        ...


    def serialize_object_ids(self, orig_ids):
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


########################### CONCATENATED DATASET ###########################

class ConcatDataset(_ConcatDatasetBase):
    """
    A dataset class to concatenate multiple sub-datasets into a single dataset
    """
    def __init__(self, datasets: List[TrainingDataset]):
        super().__init__(datasets)