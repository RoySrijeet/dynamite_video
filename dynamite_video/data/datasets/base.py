import cv2
import itertools
import math
import random

import numpy as np
import pycocotools.mask as mt

from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset as _ConcatDataset
from typing import Any, Dict, List

from dynamite_video.data.generic_video_parser import GenericVideoSequence


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
        self.sample_instance_counts = []

        self.fpack_reader = None
    

    def __len__(self):
        return len(self.samples)
    
    
    @abstractmethod
    def map_annotations(self, *args, **kwargs):
        """Map dataset-specific annotation info to a common dataset dictionary format"""
        pass

    def random_draw(self, x):
        while True:
            yield random.randint(0, x)

    def mask_area(self, rle, img_dims):
        """
        Area of an RLE segment
        """
        return mt.area({
            "counts": rle.encode("utf-8"),
            "size": img_dims
        })
    

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
                    "vid_id": vid_id,
                    "ref_frame": t,
                    "other_frames": other_frame_idxes,
                    "video": vid,
                })

                # store original clip resolution
                train_sample_dims.append((vid.height, vid.width))

        # restore initial random state
        random.setstate(rnd_state_backup)
        random.shuffle(train_samples)

        train_samples = train_samples[:num_total_samples]

        return train_samples

    
    # def create_training_samples_vis(
    #     self, 
    #     videos: Dict[str, GenericVideoSequence],
    #     num_total_samples: int,
    #     max_num_instances: int=4,
    # ):
    #     """
    #     Generate a set number of samples from the dataset

    #     Args:
    #         videos: all video sequences in the dataset as GenericVideoSequence objects
    #         num_total_samples: total num of samples to be drawn from the dataset
    #         max_num_instances: maximum #instances permitted in a clip, default: 4
    #     """

    #     # fix seed so that same set of samples is generated across all processes
    #     rnd_state_backup = random.getstate()
    #     random.seed(2202)

    #     # given the starting frame index of a clip, the rest of the frames of the clip are randomly sampled from 
    #     # a temporal window of a certain size, specified by the configurable `frame_sampling_multiplicative_factor`
    #     max_temporal_span = int(round(self.frame_sampling_multiplicative_factor * self.clip_length))

    #     samples_by_num_instance = defaultdict(list)
    #     for vid_id, vid in videos.items():
    #         # last index in the video to be the first frame of a clip
    #         last_t = len(vid) - self.clip_length

    #         # separate (video, frame index) into bins. The bins range from 1 to configurable `max_num_instances`. 
    #         # From a pair in the N-th bin, a clip can be sampled (starting at t) with up to N instances in it
    #         for t in range(last_t):
    #             # instances present in the clip starting at frame t, are the instances present in frame t
    #             valid_instance_ids = [iid for iid in vid.instance_ids if iid in vid.segmentations[t]]
    #             if not valid_instance_ids:
    #                 continue
    #             bin_id = min(max_num_instances, len(valid_instance_ids))
    #             samples_by_num_instance[bin_id].append((vid_id, t, valid_instance_ids))

    #     # random samples within each bin
    #     for bin_id in samples_by_num_instance.keys():
    #         random.shuffle(samples_by_num_instance[bin_id])
        
    #     train_samples = []
    #     train_sample_dims = []
    #     train_sample_ni = []
        
    #     # uniformly sample videos with different instance counts
    #     num_instances_per_count = int(math.ceil(num_total_samples / float(max_num_instances)))
    #     available_sample_pool = []

    #     for ni in range(max_num_instances, 0, -1):
    #         if ni not in samples_by_num_instance.keys():
    #             continue
    #         # The pool available for ni includes the pool available for (ni+1)
    #         available_sample_pool = samples_by_num_instance[ni] + available_sample_pool

    #         for ii in range(num_instances_per_count):
    #             ii = ii % len(available_sample_pool)
                
    #             # extract metadata for a training sample (i.e., a clip)
    #             # 1. the video ID to identify the source video to extract the clip from
    #             # 2. index of the frame in the video that will be the first frame of the clip - 
    #             # let's call it a reference frame
    #             # 3. IDs of the instances in this reference frame
    #             vid_id, ref_frame_idx, instance_ids = available_sample_pool[ii]

    #             assert len(instance_ids) >= ni

    #             if len(instance_ids) > ni:
    #                 # if reference frame contains more instances than permitted, take a subset
    #                 # only happens when sampling from the final bin
    #                 instance_ids = random.sample(instance_ids, ni)

    #             # extract the video info from the dataset
    #             vid = videos[vid_id]

    #             # the rest of the frames in the clip can be sampled from 
    #             # a window of the next max_temporal_span frames
    #             other_frames_list = list(
    #                 range(ref_frame_idx + 1, min(len(vid), ref_frame_idx + max_temporal_span))
    #             )

    #             assert len(other_frames_list) >= self.clip_length - 1, \
    #                 f"Something went wrong here: {ref_frame_idx}, {len(vid)}, {other_frames_list}"

    #             # from the window sample the necessary #frames to fill the clip
    #             other_frame_idxes = sorted(random.sample(other_frames_list, self.clip_length - 1))

    #             train_samples.append({
    #                 "vid_id": vid_id,
    #                 "ref_frame": ref_frame_idx,
    #                 "other_frames": other_frame_idxes,
    #                 "ref_inst_ids": instance_ids,
    #                 "video": vid,
    #             })

    #             # store original clip resolution
    #             train_sample_dims.append((vid.height, vid.width))
    #             # store bin (==#instances in the clip) size
    #             train_sample_ni.append(ni)

    #     # restore initial random state
    #     random.setstate(rnd_state_backup)

    #     train_samples = train_samples[:num_total_samples]
    #     train_sample_dims = train_sample_dims[:num_total_samples]
    #     train_sample_ni = train_sample_ni[:num_total_samples]

    #     return train_samples, train_sample_dims, train_sample_ni

    # def __getitem__(self, index):
    #     n_tries = 0
    #     while True:

    #         sample = self.parse_sample(index)
    #         if sample is not None:
    #             return sample

    #         num_instances = self.sample_instance_counts[index]
    #         self.fallback_candidates[num_instances].discard(index)
    #         index = random.choice(list(self.fallback_candidates[num_instances]))
    #         n_tries += 1
    #         if n_tries % 3 == 0:
    #             print(f"Num failed tries = {n_tries} for dataset {self.name}, num_instances {num_instances}")
    

    # def parse_sample(self, index):
    #     """
    #     Prepare a sample (training clip) for training forward pass

    #     Args:
    #         index: index of the sample to return. The index addresses a sample, 
    #             which is a dict with the following keys:
    #             `vid_id`: id of the source video
    #             `ref_frame`: first frame of the clip
    #             `other_frames`: list of rest of the frames in the clip
    #             `ref_inst_ids`: instances present in the clip

    #     """
    #     sample = self.samples[index]
    #     video = self.videos[sample['vid_id']]

    #     # get absolute frame indices in the video that will form a clip
    #     frame_indices = [sample["ref_frame"]] + sample["other_frames"]
    #     clip_inst_ids = sample["ref_inst_ids"]
    #     clip_id = sample["vid_id"]
        
    #     # extract clip from the video
    #     clip = video.extract_subsequence(frame_indices, clip_inst_ids, clip_id)

    #     # load RGB frames as a list of np.ndarrays, each [H, W, 3]
    #     images = clip.load_images()
        
    #     # load binary and semantic masks as lists
    #     masks, semantic_masks, instance_maps = clip.prepare_masks()

    #     # color augmentations
    #     images = self.apply_color_augmentation(images)

    #     images = np.stack(images)                                       # [T, H, W, 3]
    #     masks = np.stack([np.stack(masks_t) for masks_t in masks])      # [T, N, H, W]
    #     masks = np.transpose(masks, (1, 0, 2, 3))                       # [N, T, H, W]
    #     semantic_masks = np.stack(semantic_masks)                       # [T, H, W]

    #     meta_info = {
    #         "orig_dims": images[0].shape[:2],
    #         "seq_name": video.id,
    #         "frame_indices": frame_indices,
    #         "instance_maps": instance_maps,
    #     }

    #     # data augmentations
    #     # resize shortest edge
    #     images, masks, semantic_masks = self.resize_shortest_edge(images, masks, semantic_masks)
    #     # apply horizontal flip
    #     images, masks, semantic_masks = self.apply_random_horizontal_flip(images, masks, semantic_masks)
    #     # apply random crops
    #     images, masks, semantic_masks, padding_mask = self.apply_random_crop(images, masks, semantic_masks)
    #     bg_masks = np.logical_not(np.logical_or(padding_mask, semantic_masks)).astype('uint8')

    #     images = np.transpose(images, (0, 3, 1, 2))                  # [T, H, W, 3] -> [T, 3, H, W]
    #     if self.cfg.INPUT.RGB and self.cfg.INPUT.AUGMENTATION.RGB_TO_BGR:
    #         images = np.flip(images, 1).copy()                       # RGB -> BGR

    #     return {
    #         "images": images,
    #         "instance_masks": masks,
    #         "semantic_masks": semantic_masks,
    #         "padding_mask": padding_mask,
    #         "bg_masks": bg_masks,
    #         "ref_frame_index": 0,
    #         "dataset": self.name,
    #         "meta": meta_info
    #     }

    # def sample_dataset_names(self):
    #     return [self.name] * len(self.sample_image_dims)




######################### INFERENCE DATASET BASE ###################################


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
        # store instance IDs present in each frame
        instances_per_frame = []
        # for each instance, store index of the frame where it first appeared
        instance_discovery = {}
        
        for fr_idx, fp in enumerate(filepaths):
            try:
                msk = np.asarray(Image.open(fp)).astype('uint8')
            except:
                raise RuntimeError(f"Problem loading mask from {fp}")
            
            semantic_maps.append(msk)
            
            # instances present in this frame
            instances = list(np.unique(msk))[1:]
            instances_per_frame.append(instances)

            # instance discovery
            for inst_id in instances:
                if inst_id not in instance_discovery.keys():
                    instance_discovery[inst_id] = fr_idx
            
            # bg mask
            bg_msk = (msk == 0).astype('uint8')
            bg_masks.append(bg_msk)
        
        semantic_maps = np.stack(semantic_maps)
        bg_masks = np.stack(bg_masks)

        return semantic_maps, bg_masks, instances_per_frame, instance_discovery
    

    def load_rle_masks(self):
        ...


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

    def mask_area(self, rle, img_dims):
        """
        Area of an RLE segment
        """
        return mt.area({
            "counts": rle.encode("utf-8"),
            "size": img_dims
        })

    def decode_mask(self, rle, img_dims=None):
        """
        Decode RLE mask to numpy.ndarray
        """
        if img_dims is not None:
            encoded_mask = {
                "counts": rle.encode("utf-8"),
                "size": img_dims
            }
        else:
            encoded_mask = rle
        return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)


########################### CONCATENATED DATASET ###########################

class ConcatDataset(_ConcatDataset):
    """
    A dataset class to concatenate multiple sub-datasets into a single dataset
    """
    def __init__(self, datasets: List[TrainingDataset]):
        super().__init__(datasets)

        self.sample_image_dims = list(itertools.chain(*[ds.sample_image_dims for ds in self.datasets]))

    def sample_dataset_names(self):
        return list(itertools.chain(*[ds.sample_dataset_names() for ds in self.datasets]))