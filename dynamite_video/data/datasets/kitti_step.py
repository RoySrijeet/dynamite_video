import os
import json
import numpy as np
import pycocotools.mask as mt

from collections import defaultdict
from typing import Any, Dict, List, Union


from dynamite_video.utils.paths import Paths
from dynamite_video.data.datasets.base import TrainingDataset, InferenceDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks

class KITTISTEPTrainingDataset(TrainingDataset):
    """
    KITTI-STEP Training Dataset Class

    Creates a `torch.utils.data.Dataset` class to load KITTI-STEP dataset
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
        fps = cfg.DATASETS.KITTI_STEP.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.KITTI_STEP.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR
        max_num_instances = cfg.DATASETS.KITTI_STEP.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "KITTI_STEP", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to KITTI_STEP images
        path_to_images = Paths.to_kitti_step_trainval_images()
        if not os.path.exists(path_to_images):
            # `path_to_images` could be an fpack file
            path_to_images = f"{path_to_images}.fpack"
            assert os.path.exists(path_to_images), f"Directory not found: {path_to_images}"
        
        # read JSON annotations
        annotations_content = self.map_annotations(Paths.to_kitti_step_train_annotations())
        
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



    def map_annotations(self, annotations_path: str):
        """
        Read KITTI-STEP annotations from JSON file
        """

        # only allow instances with mask area above a certain threshold
        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
        # read JSON file
        with open(annotations_path, 'r') as fh:
            content = json.load(fh)

        sequences = []

        for seq in content["sequences"]:
            
            seq['id'] = f"{self.name}/{seq['id']}"
            
            # video resolution
            img_dims = (seq['height'], seq['width'])
            
            updated_segmentations = []      # store valid segmentations (binary)
            salient_classes = []            # store salient class IDs whose instances are present in the frame
            accepted_track_ids = {}         # store accepted instance IDs
            
            # salient classes ('person' and 'car') - with instance-level annotations
            for fr_idx, segs_t in enumerate(seq['segmentations']):
                
                updated_segmentations.append(dict())
                salient_classes.append(set())
                
                # add instance masks of the salient classes
                for track_id, seg in segs_t.items():
                    # only consider instances with a minimum mask area
                    if self.mask_area(seg, img_dims) >= MIN_MASK_AREA:
                        # store instance mask
                        updated_segmentations[-1][int(track_id)] = seg
                        accepted_track_ids[int(track_id)] = seq['categories'][track_id]
                        # note the salient classes present in the frame
                        salient_classes[-1].add(seq['categories'][track_id])

            # maximum instance ID (belonging only to salient classes) seen across all frames
            max_track_id = max(accepted_track_ids.keys())
            
            # panoptic masks of 'stuff' classes
            # Label values for panoptic class annotations start from max_track_id + 1 
            for fr_idx, pano_masks in enumerate(seq["semantic_segmentations"]):
                for class_id, pano_seg in pano_masks.items():
                    
                    # ignore if 'void' class
                    if class_id == '255':
                        continue
                    
                    # skip any annotation that belongs to the salient instances present in this frame
                    if int(class_id) not in salient_classes[fr_idx]:
                        if self.mask_area(pano_seg, img_dims) >= MIN_MASK_AREA:
                            # store panoptic mask with unique ID per 'stuff' class
                            stuff_track_id = max_track_id + int(class_id) + 1
                            updated_segmentations[fr_idx][stuff_track_id] = pano_seg
                            accepted_track_ids[int(stuff_track_id)] = int(class_id)

                    # TODO - some salient classes may not be fully represented in instance masks
                    # they automatically become part of the background


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

    def decode_mask(self, rle, img_dims=None):
        """
        Decode RLE mask to numpy.ndarray
        """
        encoded_mask = {
            "counts": rle.encode("utf-8"),
            "size": img_dims
        }
        return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)


class KITTISTEPInferenceDataset(InferenceDataset):
    """
    Inference dataset for KITTI-STEP

    NOTE: KITTI-STEP training dataset (21 sequences) is split into train (12) and val (9) datasets.
    Annotations for KITTI-STEP test dataset (29 different sequences) was not available
    """

    def __init__(self, cfg):
        # number of frames in each training sample
        clip_length = cfg.DATASETS.KITTI_STEP.INFERENCE.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.KITTI_STEP.INFERENCE.FPS
        # number of overlapping frames between clips
        num_overlapping_frames = cfg.DATASETS.KITTI_STEP.INFERENCE.FRAME_OVERLAP

        assert num_overlapping_frames <= clip_length, f"No. of overlapping frames cannot be more than the length of a clip"
        
        split = cfg.DATASETS.KITTI_STEP.INFERENCE.SPLIT
        
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
            self.annotations = val_anno
            
            del trainval_annotations

        else:
            assert self.split == "test"
            raise RuntimeError(f"Annotations for KITTI-STEP test split is not available")
            # get paths
            self.path_to_images = Paths.to_kitti_step_test_images()
            if not os.path.exists(path_to_images):
                # `path_to_images` could be an fpack file
                path_to_images = f"{path_to_images}.fpack"
                assert os.path.exists(path_to_images), f"Directory not found: {path_to_images}"
            
            # JSON annotation file
            # N/A
    
    def create_inference_dataset(self):
        """
        Prepare dataset for evaluation.
        """

        if self.split != "val":
            raise RuntimeError(f"Annotations for KITTI-STEP {self.split} split is not available")
        
        sequence_annotations = []
        
        for seq in self.annotations["sequences"]:

            metadata = {"id": seq['id']}

            # load images
            image_filepaths = sorted([os.path.join(self.path_to_images, seq['id'], file) for file in os.listdir(os.path.join(self.path_to_images, seq['id'])) if file.endswith('png')])
            images = self.load_images(image_filepaths)      # [T, H, W, 3]

            metadata["length"] = len(image_filepaths)
            metadata["orig_dims"] = (images.shape[1], images.shape[2])

            # TODO: excuse the lingo, the following 'instances' are not really instances, rather classes
            
            # for each instance (class), store index of the frame where it first appeared
            instance_discovery = {}
            
            for fr_idx, pano_masks in enumerate(seq["semantic_segmentations"]):
                for class_id, _ in pano_masks.items():
                    # ignore void class
                    if class_id == "255": 
                        continue
                    if int(class_id) not in instance_discovery.keys():
                        # encountered an instance (class) for the first time in the sequence
                        instance_discovery[int(class_id)] = fr_idx
                    
            
            # find all the instances (classes) present in the sequence, so that we can insert empty
            # masks for the ones that are absent
            orig_instance_ids = sorted(list(instance_discovery.keys()))
            orig_to_serial_ids, serial_to_orig_ids = self.serialize_instance_ids(orig_instance_ids)
            assert orig_instance_ids == list(orig_to_serial_ids.keys())
            metadata["orig_to_serial_ids"] = orig_to_serial_ids
            metadata["serial_to_orig_ids"] = serial_to_orig_ids

            metadata["instance_discovery"] = {orig_to_serial_ids[inst_id]: fr_idx 
                                              for inst_id, fr_idx in instance_discovery.items()}

            # load panoptic masks
            instance_masks = []
            semantic_maps = []
            bg_masks = []
            # store instance (class) IDs present in each frame
            instances_per_frame = []
            for fr_idx, pano_masks in enumerate(seq["semantic_segmentations"]):
                
                fr_instance_masks = []
                fr_semantic_map = np.zeros(metadata["orig_dims"])
                fr_instance_ids = []
                
                # look for a mask of each instance (class) in each frame
                for inst_id in orig_instance_ids:
                    
                    if str(inst_id) in pano_masks.keys():
                        # if found, decode the RLE mask into an np.ndarray
                        m = self.decode_mask(pano_masks[str(inst_id)], metadata["orig_dims"])
                        fr_instance_masks.append(m)
                        # also save it in a semantic map, with serialized ID
                        fr_semantic_map[np.where(m==1)] = orig_to_serial_ids[inst_id]
                        fr_instance_ids.append(orig_to_serial_ids[inst_id])
                    else:
                        # if not found, use an empty mask
                        fr_instance_masks.append(np.zeros_like(fr_semantic_map).astype(np.uint8))
                
                instance_masks.append(np.stack(fr_instance_masks))
                semantic_maps.append(fr_semantic_map)
                fr_bg_mask = (fr_semantic_map==0).astype(np.uint8)
                bg_masks.append(fr_bg_mask)
                instances_per_frame.append(fr_instance_ids)

            instance_masks = np.stack(instance_masks)               # T, N, H, W
            semantic_maps = np.stack(semantic_maps)                 # T, H, W
            bg_masks = np.stack(bg_masks)                           # T, H, W

            if self.cfg.INPUT.AUGMENTATION.RESIZE_TEST:
                # compute target resolution
                new_height, new_width = compute_resized_dims(
                    *images.shape[1:3], 
                    min_dim=self.cfg.INPUT.AUGMENTATION.MIN_DIM_TEST,
                    max_dim=self.cfg.INPUT.AUGMENTATION.MAX_DIM_TEST,
                )
                if (new_height, new_width) != metadata["orig_dims"]:
                    images = resize_images(images, new_height, new_width)
                    semantic_maps = resize_masks(semantic_maps, new_height, new_width, binary=False)
                    instance_masks = resize_masks(instance_masks, new_height, new_width, binary=True)
                    bg_masks = resize_masks(bg_masks, new_height, new_width, binary=False)
            
            # arrange dimensions
            images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
            if self.cfg.INPUT.RGB:
                # BGR -> RGB (load_images uses cv2.imread which reads images in BGR mode by default)
                images = np.flip(images, 1).copy()
            
            metadata["images"] = images
            metadata["bg_masks"] = bg_masks
            metadata["semantic_maps"] = semantic_maps
            metadata["instance_masks"] = instance_masks

            # TODO - padding - not applied
            metadata["padding_mask"] = np.zeros((images.shape[2], images.shape[3])).astype('uint8')

            metadata["instances_per_frame"] = instances_per_frame
            metadata["clip_length"] = self.clip_length
            metadata["num_overlapping_frames"] = self.num_overlapping_frames
            
            sequence_annotations.append(metadata)
            # TODO: remove
            break
        
        return sequence_annotations