import os
import json
import numpy as np
import pycocotools.mask as mt

from PIL import Image
from collections import defaultdict
from typing import Any, Dict, List, Union


from dynamite_video.utils.paths import Paths
from dynamite_video.data.datasets.base import TrainingDataset, InferenceDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset

class VIPSEGTrainingDataset(TrainingDataset):
    """
    VIPSEG Training Dataset Class

    Creates a `torch.utils.data.Dataset` class to load VIPSEG dataset
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
        fps = cfg.DATASETS.VIPSEG.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.VIPSEG.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR
        max_num_instances = cfg.DATASETS.VIPSEG.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "VIPSEG", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to VIPSEG images
        path_to_images = Paths.to_vipseg_images()
        # if not os.path.exists(path_to_images):
        #     # `path_to_images` could be an fpack file
        #     path_to_images = f"{path_to_images}.fpack"
        #     assert os.path.exists(path_to_images), f"Directory not found: {path_to_images}"
        
        # path to VIPSEG annotations
        # path_to_annotations = Paths.to_vipseg_annotations()
        # if not os.path.exists(path_to_annotations):
        #     # `path_to_images` could be an fpack file
        #     path_to_annotations = f"{path_to_annotations}.fpack"
        #     assert os.path.exists(path_to_annotations), f"Directory not found: {path_to_annotations}"

        # training video info
        
        annotations_content = self.map_annotations(path_to_images, 
                                                   Paths.to_vipseg_annotations(), 
                                                   Paths.to_vipseg_train_video_info()
                                            )
        
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



    def map_annotations(
            self, 
            path_to_images, 
            path_to_annotations: str, 
            video_info_path: str
    ):
        """
        Read VIPSEG annotations
        """

        # only allow instances with mask area above a certain threshold
        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
        with open(video_info_path, "r") as f:
            video_info = json.load(f)

        # Convention: For panoptic masks, category IDs range from 0 to 124. "0" denotes VOID class. 
        # For "stuff" classes, the value of masks is the same as the category ID. 
        # For "thing" classes, the value of masks is "category_id x 100 + instance_id". 
        # Example - values of masks of "person" (category ID = 61) instances are "6100", "6101", ...
        # Thus, instances with mask values larger than 124 belong to "thing" classes, rest to "stuff".

        # NOTE: In JSON file, category IDs range from 0-123 and excludes VOID class. However, the values 
        # in the masks still correspond to original convention. So, the category IDs are shifted to match
        # the original convention.
        
        # store category labels in meta info
        categories = {}
        # keep a separate record of IDs of "thing" classes
        isthing = []
        for entry in video_info["categories"]:
            categories[int(entry['id'])+1] = entry['name']
            if entry["isthing"]:
                isthing.append(int(entry['id'])+1)
        meta_info = {
            "category_labels": categories,
            "thing_ids": isthing
        }
        
        # read annotations
        sequence_annotations = []
        for seq in video_info["sequences"]:
            
            entry = {}
            entry["id"] = f"{self.name}/{seq['name']}"
            entry["dataset"]  = self.name
            entry["height"] = seq["height"]
            entry["width"] = seq["width"]
            
            # path to image files
            entry["image_paths"] = sorted([os.path.join(path_to_images, seq["name"], file + '.jpg') for file in seq["filenames"]])
            # mask files
            maskfiles = sorted([os.path.join(path_to_annotations, seq["name"], file + '.png') for file in seq["filenames"]])
            
            segmentations = []
            accepted_stuff_classes = set()
            accepted_track_ids = set()
            for idx, file in enumerate(maskfiles):
                # read panoptic mask
                mask = np.asarray(Image.open(file))
                
                # instances of both "thing" and "stuff" classes
                instances = list(np.unique(mask))[1:]       # ignore VOID (0)

                binary_masks = {}
                for inst_id in instances:
                    # extract binary mask
                    _m = (mask==inst_id).astype(dtype='uint8')
                    # check mask area
                    if self.mask_area(_m) >= MIN_MASK_AREA:
                        # encode mask array as RLE
                        binary_masks[int(inst_id)] = mt.encode(np.asfortranarray(_m))
                        if (inst_id-1) in seq["stuff_classes"]:
                            accepted_stuff_classes.add(int(inst_id))
                        else:
                            # instance belongs to a "thing" class
                            accepted_track_ids.add(int(inst_id))

                segmentations.append(binary_masks)
            
            entry["segmentations"] = segmentations
            entry["stuff_classes"] = list(accepted_stuff_classes)
            # assigned_id = category_id x 100 + instance_id
            entry["thing_classes"] = [k//100 for k in accepted_track_ids]
            entry["categories"] = {k:k//100 for k in accepted_track_ids} # of "thing" classes
            entry["categories"].update({k:k for k in accepted_stuff_classes})

            sequence_annotations.append(entry)
            
            # # TODO - remove
            if len(sequence_annotations) == 30:
                break

        return {
            "sequences": sequence_annotations,
            "meta": meta_info
        }

    def mask_area(self, mask):
        assert isinstance(mask, np.ndarray)
        bin_mask = mask.astype('uint8')
        assert list(np.unique(bin_mask))==[0,1]
        return bin_mask.sum()



class VIPSEGInferenceDataset(InferenceDataset):
    ...

