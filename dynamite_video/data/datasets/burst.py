import os
import json
import pycocotools.mask as mt

from collections import defaultdict
from typing import Dict

from dynamite_video.data.datasets.base import TrainingDataset, InferenceDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths


########################### TRAINING DATASET ###########################

class BURSTTrainingDataset(TrainingDataset):
    """BURST Training Dataset Class"""
    
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
        fps = cfg.DATASETS.BURST.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.BURST.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR
        max_num_instances = cfg.DATASETS.BURST.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "BURST", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to BURST images
        self.path_to_images = Paths.to_burst_train_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/burst.fpack"
            assert os.path.exists(path_to_images), f"BURST images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)
        
        # read JSON annotations
        annotations_content = self.map_annotations(Paths.to_burst_training_annotations())
        
        # cast each video sequence in the dataset to a generic `GenericVideoSequence` template
        videos, meta_info = parse_generic_video_dataset(self.path_to_images, annotations_content)
        
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
        Read BURST annotations from JSON file
        """

        # only allow instances with mask area above a certain threshold
        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
        # read JSON file
        with open(annotations_path, 'r') as fh:
            content = json.load(fh)

        sequences = []

        for seq in content["sequences"]:
            
            # video resolution
            img_dims = (seq['height'], seq['width'])
            
            # video ID (sub-dataset/sequence name)
            seq["id"] = f"{self.name}/{seq['dataset']}/{seq['seq_name']}"
            
            # relative paths to all JPEG images of the dataset (only the annotated ones)
            seq["image_paths"] = [f"{seq['dataset']}/{seq['seq_name']}/{fname}" for fname in seq["annotated_image_paths"]]
            
            updated_segmentations = []
            accepted_track_ids = set()
            # filter out small instances
            for fr_idx, segs_t in enumerate(seq['segmentations']):
                updated_segmentations.append(dict())
                for track_id, seg in segs_t.items():
                    # only consider instances with a minimum mask area
                    if self.mask_area(seg['rle'], img_dims) >= MIN_MASK_AREA:
                        # store instance mask
                        updated_segmentations[-1][int(track_id)] = seg['rle']
                        accepted_track_ids.add(track_id)

            # if none of the instances are large enough, exclude this sequence
            if not accepted_track_ids:
                continue

            seq['segmentations'] = updated_segmentations
            seq["categories"] = {
                int(track_id): seq["track_category_ids"][track_id]
                for track_id in accepted_track_ids
            }

            # remove unnecessary fields
            FIELDS_TO_DELETE = ["seq_name", "neg_category_ids", "not_exhaustive_category_ids", "track_category_ids", "all_image_paths", "annotated_image_paths"]
            for field in FIELDS_TO_DELETE:
                del seq[field]

            sequences.append(seq)

        # store category id to name mapping
        meta_info = {
            "category_labels": {
                int(cat['id']): cat['name'] for cat in content["categories"]
            }
        }
        del content

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


class BURSTInferenceDataset(InferenceDataset):
    ...

