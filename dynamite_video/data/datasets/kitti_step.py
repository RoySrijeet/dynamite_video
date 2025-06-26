import os
import json
import numpy as np
import pycocotools.mask as mt

from typing import Dict

from dynamite_video.data.datasets.base import TrainingDataset, EvaluationDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.data_utils import decode_mask, mask_area
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths


def map_kitti_step_annotations(cfg, content, MIN_MASK_AREA):
    """
    Read KITTI-STEP evaluation annotations from JSON

    Returns a dictionary with annotation content from the entire dataset
    """
    sequences = []
    
    max_instances_per_category = cfg.DATASETS.KITTI_STEP.MAX_INSTANCES_PER_CATEGORY
    ignore_class = cfg.DATASETS.KITTI_STEP.IGNORE_CLASS
    thing_classes = cfg.DATASETS.KITTI_STEP.THINGS_CLASSES

    # NOTE: the semantic classes in KITTI_STEP have IDs from 0-18 and 255 (void).
    # The instance segmentations have independent IDs that overlap with the
    # semantic class IDs. To resolve this issue,
    # semantic classes are assigned new ID: class ID * max_instances_per_category
    # instances are assigned new ID: class ID * max_instances_per_category + instance ID
    
    for seq in content["sequences"]:

        seq["dataset"] = "KITTI_STEP"

        # video resolution
        img_dims = (seq["height"], seq["width"])
        
        updated_segmentations = []
        accepted_track_ids = {}
        ignore_masks = []
        
        # read semantic maps
        for fr_idx, sem_masks in enumerate(seq["semantic_segmentations"]):
            updated_segmentations.append(dict())
            ignore_masks.append([])

            for class_id, sem_seg_rle in sem_masks.items():

                # ignore 'void' class and 'thing' classes
                if int(class_id) == ignore_class or class_id in thing_classes:
                    ignore_masks[-1].append(decode_mask(sem_seg_rle, img_dims))
                    continue
                
                if mask_area(sem_seg_rle, img_dims) >= MIN_MASK_AREA:
                    track_id = int(class_id) * max_instances_per_category
                    updated_segmentations[-1][track_id] = sem_seg_rle
                    accepted_track_ids[track_id] = int(class_id)

            if len(ignore_masks[-1]) == 0:
                # if no void mask found, add an empty one
                ignore_masks[-1] = np.zeros(img_dims).astype(np.uint8)
            else:
                ignore_masks[-1] = np.any(np.stack(ignore_masks[-1]), axis=0).astype(np.uint8)
        
        # read instance masks
        for fr_idx, fr_inst_masks in enumerate(seq["segmentations"]):
            for track_id, inst_rle in fr_inst_masks.items():

                if mask_area(inst_rle, img_dims) >= MIN_MASK_AREA:
                    # id of the class the instance belongs to
                    class_id = seq['categories'][track_id]
                    # new track ID
                    new_track_id = class_id * max_instances_per_category + int(track_id)
                    updated_segmentations[fr_idx][new_track_id] = inst_rle
                    accepted_track_ids[new_track_id] = seq['categories'][track_id]
        

        seq["segmentations"] = updated_segmentations
        seq["ignore_masks"] = [mt.encode(np.asfortranarray(ig_msk))["counts"].decode('utf-8') for ig_msk in ignore_masks]
        seq["categories"] = accepted_track_ids

        seq.pop("semantic_segmentations")
        sequences.append(seq)
    
    # store category id to name mapping
    meta_info = content["meta"]["category_labels"]
    meta_info = {
        "category_labels": {
            int(id): name for id, name in content["meta"]["category_labels"].items()
        },
        "num_classes": cfg.DATASETS.KITTI_STEP.NUM_CLASSES,
        "things_list": cfg.DATASETS.KITTI_STEP.THINGS_CLASSES,
        "ignore_class": cfg.DATASETS.KITTI_STEP.IGNORE_CLASS,
        "max_instances_per_category": cfg.DATASETS.KITTI_STEP.MAX_INSTANCES_PER_CATEGORY,
        "min_mask_area": MIN_MASK_AREA,
    }

    return {
        "sequences": sequences,
        "meta": meta_info
    }


########################### TRAINING DATASET ###########################

class KITTISTEPTrainingDataset(TrainingDataset):
    """KITTI-STEP Training Dataset Class"""
    
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
        fps = cfg.DATASETS.KITTI_STEP.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.KITTI_STEP.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR

        super().__init__(cfg, "KITTI_STEP", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to KITTI_STEP images
        self.path_to_images = Paths.to_kitti_step_trainval_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/kitti_step.fpack"
            assert os.path.exists(path_to_images), f"KITTI_STEP images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)
        
        # load JSON file with annotations
        with open(Paths.to_kitti_step_train_annotations(), 'r') as fh:
            json_annotations = json.load(fh)
        
        # read annotations
        annotations_content = map_kitti_step_annotations(cfg, json_annotations, MIN_MASK_AREA=cfg.TRAINING.MIN_MASK_AREA)

        # cast each video sequence in the dataset to a generic `GenericVideoSequence` template
        videos, meta_info = parse_generic_video_dataset(self.path_to_images, annotations_content)
        
        self.meta = meta_info
        self.videos: Dict[str, GenericVideoSequence] = {vid.id: vid for vid in videos}

        # create samples
        self.samples = self.create_training_samples(self.videos, num_samples)

        self.sample_image_dims = [[1, 1] for _ in range(num_samples)]



########################### EVALUATION DATASET ###########################


class KITTISTEPEvaluationDataset(EvaluationDataset):
    """
    Evaluation dataset for KITTI_STEP

    NOTE: KITTI-STEP training dataset (21 sequences) is split into train (12) and val (9) datasets.
    Annotations for KITTI-STEP test dataset (29 different sequences) was not available
    """

    def __init__(self, cfg):
        # number of frames in each training sample
        clip_length = cfg.DATASETS.KITTI_STEP.EVALUATION.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.KITTI_STEP.EVALUATION.FPS
        # number of overlapping frames between clips
        num_overlapping_frames = cfg.DATASETS.KITTI_STEP.EVALUATION.FRAME_OVERLAP

        assert num_overlapping_frames <= clip_length, f"No. of overlapping frames cannot be more than the length of a clip"
        
        split = cfg.DATASETS.KITTI_STEP.EVALUATION.SPLIT
        
        super().__init__(cfg, "KITTI_STEP", clip_length, fps, num_overlapping_frames, split)

        if self.split == "val":
            # get paths
            self.path_to_images = Paths.to_kitti_step_trainval_images()
            if not os.path.exists(self.path_to_images):
                # `path_to_images` could be an fpack file
                self.path_to_images = f"{self.path_to_images}.fpack"
                assert os.path.exists(self.path_to_images), f"Directory not found: {self.path_to_images}"
            
            # get annotations
            train_annotations_json = Paths.to_kitti_step_train_annotations()
            with open(train_annotations_json, "rb") as f:
                train_annotations = json.load(f)
            f.close()
            train_ids = []
            for i in range(len(train_annotations["sequences"])):
                train_ids.append(train_annotations["sequences"][i]["id"])
            del train_annotations

            trainval_annotations_json = Paths.to_kitti_step_trainval_annotations()
            with open(trainval_annotations_json, "rb") as f:
                trainval_annotations = json.load(f)
            f.close()
            val_anno = {"sequences": []}
            for i in range(len(trainval_annotations["sequences"])):
                seq_id = trainval_annotations["sequences"][i]["id"]
                if seq_id in train_ids:
                    continue
                val_anno["sequences"].append(trainval_annotations["sequences"][i])
            val_anno["meta"] = trainval_annotations["meta"]
            json_annotations = val_anno
            
            del trainval_annotations

        else:
            assert self.split == "test"
            raise RuntimeError(f"Annotations for KITTI-STEP test split is not available")

        annotations_content = map_kitti_step_annotations(cfg, json_annotations, MIN_MASK_AREA=1)

        self.videos, self.meta = parse_generic_video_dataset(self.path_to_images, annotations_content) #, serialize=True)
        del annotations_content
        
        self.meta["clip_length"] = self.clip_length
        self.meta["num_overlapping_frames"] = self.num_overlapping_frames
        self.meta["fps"] = self.fps
        self.meta["split"] = self.split