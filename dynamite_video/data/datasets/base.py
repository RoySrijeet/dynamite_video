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
        self.sample_object_counts = []

        self.fpack_reader = None
    

    def __len__(self):
        return len(self.samples)

    def random_draw(self, x):
        while True:
            yield random.randint(0, x)
    
    
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

class ConcatDataset(_ConcatDataset):
    """
    A dataset class to concatenate multiple sub-datasets into a single dataset
    """
    def __init__(self, datasets: List[TrainingDataset]):
        super().__init__(datasets)

        self.sample_image_dims = list(itertools.chain(*[ds.sample_image_dims for ds in self.datasets]))

    def sample_dataset_names(self):
        return list(itertools.chain(*[ds.sample_dataset_names() for ds in self.datasets]))