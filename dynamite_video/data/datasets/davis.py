import os
import numpy as np
import pycocotools.mask as mt

from PIL import Image
from typing import Any, Dict, List
from collections import defaultdict

from utils.paths import Paths
from data.datasets.base import TrainingDataset, InferenceDataset
from data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset


class DAVISTrainingDataset(TrainingDataset):
    """
    DAVIS Training Dataset Class

    Creates a `torch.utils.data.Dataset` class to load BURST dataset
    """
    
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
        fps = cfg.DATASETS.DAVIS.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.DAVIS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR
        max_num_instances = cfg.DATASETS.DAVIS.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "DAVIS", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # get paths
        path_to_images = Paths.to_davis_images()
        # `images_dir` could be an fpack file
        if not os.path.exists(path_to_images):
            path_to_images = f"{path_to_images}.fpack"
            assert os.path.exists(path_to_images), f"Directory not found: {path_to_images}"
        

        # load masks of video frames in DAVIS training split 
        annotations_content = self.map_annotations(path_to_images, Paths.to_davis_annotations(), Paths.to_davis_train_imset())
        
        # cast each video sequence in the dataset to a generic `GenericVideoSequence` template
        videos, meta_info = parse_generic_video_dataset(path_to_images, annotations_content)
        
        self.meta = meta_info
        self.videos: Dict[str, GenericVideoSequence] = {vid.id: vid for vid in videos}

        # create samples
        self.samples, self.sample_image_dims, self.sample_instance_counts = self.create_training_samples(
            self.videos, num_samples, frame_sampling_multiplicative_factor, max_num_instances
        )
        
        # store fallback candidates
        self.fallback_candidates = defaultdict(set)
        for i, num_instances in enumerate(self.sample_instance_counts):
            self.fallback_candidates[num_instances].add(i)


    # dynamite style
    def map_annotations(
            self,
            path_to_images: str,
            path_to_annotations: str, 
            path_to_imset: str, 
    ):
        """
        Read semantic masks from PNG files

        Args:
            path_to_images: path to image directory
            path_to_annotations: path to annotations directory
            path_to_imset: path to imset .txt file, listing training sequence names

        Returns a dictionary with annotation content from the entire dataset
        """
        
        # load the list of training sequences as a list
        with open(path_to_imset, 'r') as f:
            sequences = [seq.rstrip() for seq in f.readlines()]

        sequence_annotations = []

        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
        # for each video sequence in the dataset
        for seq in sequences:
            entry = {}
            entry["id"] = f"{self.name}/{seq}"
            entry["dataset"] = self.name

            # load paths to all image files of the sequence
            imagefiles = sorted([os.path.join(path_to_images, seq, file) for file in os.listdir(os.path.join(path_to_images, seq)) if file.endswith('jpg')])
            entry["image_paths"] = imagefiles
            
            # load paths to all mask files of a sequence
            maskfiles = sorted([os.path.join(path_to_annotations, seq, file) for file in os.listdir(os.path.join(path_to_annotations, seq)) if file.endswith('png')])

            # read first frame mask again to store resolution and instances
            mask0 = np.asarray(Image.open(maskfiles[0]))
            entry["height"], entry["width"] = mask0.shape

            segmentations = []
            seq_instances = []
            # for mask of each frame of the video sequence
            for idx, file in enumerate(maskfiles):
                # read and store the semantic map
                mask = np.asarray(Image.open(file).convert("P")).astype(dtype='uint8')
                # find how many instances in the mask (excluding bg, value 0)
                instances = list(np.unique(mask))[1:]

                # extract and store binary masks of individual instances from semantic mask
                binary_masks = {}
                for i in instances:
                    _m = (mask==i).astype(dtype='uint8')
                    # check mask area
                    if self.mask_area(_m) >= MIN_MASK_AREA:
                        # if mask larger than threshold, keep it
                        binary_masks[int(i)] = mt.encode(np.asfortranarray(_m))
                        seq_instances.append(i)
                segmentations.append(binary_masks)

            seq_instances = set(seq_instances)
            entry["categories"] = {int(k):1 for k in seq_instances}
            entry['segmentations'] = segmentations

            sequence_annotations.append(entry)

            # TODO - remove
            if len(sequence_annotations) == 2:
                break

        annotations_content = {}
        # there is no explicit categories present in DAVIS
        annotations_content["meta"] = {"category_labels": {1: 'object'}}
        annotations_content["sequences"] = sequence_annotations

        return annotations_content

    def mask_area(self, mask):
        assert isinstance(mask, np.ndarray)
        bin_mask = mask.astype('uint8')
        assert list(np.unique(bin_mask))==[0,1]
        return bin_mask.sum()


class DAVISInferenceDataset(InferenceDataset):
    ...
