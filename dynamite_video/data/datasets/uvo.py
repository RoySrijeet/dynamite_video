import os
import json
import numpy as np
import pycocotools.mask as mt

from collections import defaultdict
from PIL import Image
from typing import Dict

from dynamite_video.data.datasets.base import TrainingDataset, EvaluationDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths


########################### TRAINING DATASET ###########################

class UVOTrainingDataset(TrainingDataset):
    """UVO Training Dataset Class"""
    
    def __init__(self, cfg, num_samples: int):
        """
        Read the video frames and their annotations from disc

        Args:
            cfg: configuration
            num_samples: number of samples to be used from this dataset
        """

        # number of frames in each training sample
        clip_length = cfg.TRAINING.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.UVO.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.UVO.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR
        max_num_instances = cfg.DATASETS.UVO.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "UVO", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # get paths
        self.path_to_images = Paths.to_uvo_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/uvo.fpack"
            assert os.path.exists(path_to_images), f"UVO images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)
        
        # load masks of video frames in UVO training split
        annotations_content = self.map_annotations(Paths.to_uvo_train_annotations_json())
        
        # cast each video sequence in the dataset to a generic `GenericVideoSequence` template
        videos, meta_info = parse_generic_video_dataset(self.path_to_images, annotations_content)
        
        self.meta = meta_info
        self.videos: Dict[str, GenericVideoSequence] = {vid.id: vid for vid in videos}

        # create samples
        self.samples, self.sample_image_dims, self.sample_instance_counts = self.create_training_samples(
            self.videos, num_samples, max_num_instances
        )
        
        # store fallback candidates
        self.fallback_candidates = defaultdict(set)
        for i, num_instances in enumerate(self.sample_instance_counts):
            self.fallback_candidates[num_instances].add(i)


    def map_annotations(
            self,
            path_to_annotations: str, 
    ):
        """
        Read semantic masks from PNG files

        Args:
            path_to_annotations: path to JSON annotations

        Returns a dictionary with annotation content from the entire dataset
        """
        ...
        