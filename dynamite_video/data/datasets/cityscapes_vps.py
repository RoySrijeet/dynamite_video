import os
import json
import numpy as np
import pycocotools.mask as mt

from collections import defaultdict
from PIL import Image
from typing import Dict

from dynamite_video.data.datasets.base import TrainingDataset, InferenceDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths


########################### TRAINING DATASET ###########################

class CITYSCAPESVPSTrainingDataset(TrainingDataset):
    """CITYSCAPES-VPS Training Dataset Class"""
    
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
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/cityscapes_vps.fpack"
            assert os.path.exists(path_to_images), f"CITYSCAPES_VPS images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)
        
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
        
        # read JSON file
        with open(annotations_path, 'r') as fh:
            content = json.load(fh)

        sequences = []

        # only allow instances with mask area above a certain threshold
        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA

        for seq in content["sequences"]:
            
            seq['id'] = f"{self.name}/{seq['id']}"
            
            # video resolution
            img_dims = (seq['height'], seq['width'])
            
            updated_segmentations = []      # store valid segmentations (binary)
            salient_classes = []            # store salient class IDs whose instances are present in the frame
            accepted_track_ids = {}         # store accepted instance IDs
            
            
            # filter out small instances
            for fr_idx, segs_t in enumerate(seq['segmentations']):

                updated_segmentations.append(dict())
                salient_classes.append(set())
                
                # add instance masks of the salient classes
                for track_id, seg in segs_t.items():
                    # only consider instances with a minimum mask area
                    if self.mask_area(seg, img_dims) >= MIN_MASK_AREA:
                        updated_segmentations[-1][int(track_id)] = seg
                        accepted_track_ids[int(track_id)] = seq['categories'][track_id]
                        # note the salient classes present in the frame
                        salient_classes[-1].add(seq['categories'][track_id])

            # panoptic masks
            for fr_idx, pano_masks in enumerate(seq["semantic_segmentations"]):
                for class_id, pano_seg in pano_masks.items():
                    
                    # ignore if 'void' class
                    if class_id == '255':
                        continue
                    
                    # skip any annotation that belongs to the salient instances present in this frame
                    if int(class_id) not in salient_classes[fr_idx]:
                        if self.mask_area(pano_seg, img_dims) >= MIN_MASK_AREA:
                            # store panoptic mask with unique ID per 'stuff' class
                            stuff_track_id = int(class_id)
                            updated_segmentations[fr_idx][stuff_track_id] = pano_seg
                            accepted_track_ids[int(stuff_track_id)] = int(class_id)
            
            
            # if none of the instances are large enough, exclude this sequence
            if not accepted_track_ids:
                continue
            

            seq['segmentations'] = updated_segmentations
            seq["categories"] = accepted_track_ids

            # save semantic segmentations as panoptic segmentation
            seq["panoptic_segmentations"] = seq["semantic_segmentations"]
            seq.pop("semantic_segmentations")
            sequences.append(seq)

        # store category id to name mapping
        meta_info = content["meta"]["category_labels"]
        meta_info = {
            "category_labels": {
                int(id): name for id, name in content["meta"]["category_labels"].items()
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


########################### INFERENCE DATASET ###########################


class CITYSCAPESVPSInferenceDataset(InferenceDataset):
    """
    Inference dataset for CITYSCAPES_VPS
    """

    def __init__(self, cfg):
        # number of frames in each training sample
        clip_length = cfg.DATASETS.CITYSCAPES_VPS.INFERENCE.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.CITYSCAPES_VPS.INFERENCE.FPS
        # number of overlapping frames between clips
        num_overlapping_frames = cfg.DATASETS.CITYSCAPES_VPS.INFERENCE.FRAME_OVERLAP

        assert num_overlapping_frames <= clip_length, f"No. of overlapping frames cannot be more than the length of a clip"
        
        split = cfg.DATASETS.CITYSCAPES_VPS.INFERENCE.SPLIT
        
        super().__init__(cfg, "CITYSCAPES_VPS", clip_length, fps, num_overlapping_frames, split)