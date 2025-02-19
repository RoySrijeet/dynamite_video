import os
import json
import numpy as np
import pycocotools.mask as mt

from collections import defaultdict
from typing import Any, Dict, List, Union


from dynamite_video.utils.paths import Paths
from dynamite_video.data.datasets.base import TrainingDataset, InferenceDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset

class CITYSCAPESVPSTrainingDataset(TrainingDataset):
    """
    CITYSCAPES-VPS Training Dataset Class

    Creates a `torch.utils.data.Dataset` class to load CITYSCAPES-VPS dataset
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
        fps = cfg.DATASETS.CITYSCAPES_VPS.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.CITYSCAPES_VPS.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR
        max_num_instances = cfg.DATASETS.CITYSCAPES_VPS.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "CITYSCAPES_VPS", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to CITYSCAPES_VPS images
        path_to_images = Paths.to_cityscapes_vps_train_images()
        if not os.path.exists(path_to_images):
            # `path_to_images` could be an fpack file
            path_to_images = f"{path_to_images}.fpack"
            assert os.path.exists(path_to_images), f"Directory not found: {path_to_images}"
        
        # read JSON annotations
        annotations_json_content = self.map_annotations(Paths.to_cityscapes_vps_train_annotations())
        
        # cast each video sequence in the dataset to a generic `GenericVideoSequence` template
        videos, meta_info = parse_generic_video_dataset(path_to_images, annotations_json_content)
        
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



    def map_annotations(self, annotations_path: str):
        """
        Read CITYSCAPES-VPS annotations from JSON file
        """

        # only allow instances with mask area above a certain threshold
        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
        # read JSON file
        with open(annotations_path, 'r') as fh:
            json_content = json.load(fh)

        sequences = []

        for seq in json_content["sequences"]:
            
            seq['id'] = f"{self.name}/{seq['id']}"
            
            # video resolution
            img_dims = (seq['height'], seq['width'])
            
            updated_segmentations = []
            accepted_track_ids = set()
            # filter out small instances
            for fr_idx, segs_t in enumerate(seq['segmentations']):
                updated_segmentations.append(dict())
                for track_id, seg in segs_t.items():
                    # only consider instances with a minimum mask area
                    if self.mask_area(seg, img_dims) >= MIN_MASK_AREA:
                        # store instance mask
                        updated_segmentations[-1][int(track_id)] = seg
                        accepted_track_ids.add(track_id)

            # if none of the instances are large enough, exclude this sequence
            if not accepted_track_ids:
                continue

            seq['segmentations'] = updated_segmentations
            accepted_categories = {
                int(track_id): seq["categories"][track_id]
                for track_id in accepted_track_ids
            }
            seq["categories"] = accepted_categories
            
            # save semantic segmentations as panoptic segmentation
            seq["panoptic_segmentations"] = seq["semantic_segmentations"]
            seq.pop("semantic_segmentations")
            sequences.append(seq)

            # TODO - remove
            if len(sequences) == 2:
                break

        # store category id to name mapping
        meta_info = json_content["meta"]["category_labels"]
        meta_info = {
            "category_labels": {
                int(id): name for id, name in json_content["meta"]["category_labels"].items()
            }
        }

        return {
            "sequences": sequences,
            "meta": meta_info
        }

    def mask_area(self, rle, img_dims):
        """
        Area of an RLE segment
        """
        return mt.area({
            "counts": rle.encode("utf-8"),
            "size": img_dims
        })



class CITYSCAPESVPSInferenceDataset(InferenceDataset):
    ...

