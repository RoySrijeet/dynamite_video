import os
import json
import numpy as np
import pycocotools.mask as mt

from collections import defaultdict
from typing import Dict

from dynamite_video.data.datasets.base import TrainingDataset, InferenceDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths


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
        max_num_instances = cfg.DATASETS.KITTI_STEP.TRAINING.MAX_NUM_INSTANCES

        super().__init__(cfg, "KITTI_STEP", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to KITTI_STEP images
        self.path_to_images = Paths.to_kitti_step_trainval_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/kitti_step.fpack"
            assert os.path.exists(path_to_images), f"KITTI_STEP images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)
        
        # read JSON annotations
        annotations_content = self.map_annotations(Paths.to_kitti_step_train_annotations())

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
            annotations_path: str
    ):
        """
        Read KITTI-STEP annotations from JSON file

        Args:
            path_to_annotations: path to JSON annotations

        Returns a dictionary with annotation content from the entire dataset
        """
        
        # read JSON file
        with open(annotations_path, 'r') as fh:
            content = json.load(fh)

        sequences = []

        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA # no filtering applied

        for seq in content["sequences"]:
            
            seq['id'] = f"{self.name}/{seq['id']}"
            seq['dataset'] = self.name
            
            # video resolution
            img_dims = (seq['height'], seq['width'])
            
            updated_segmentations = []
            accepted_track_ids = {}

            # read semantic maps
            for fr_idx, sem_masks in enumerate(seq["semantic_segmentations"]):
                updated_segmentations.append(dict())

                for class_id, sem_seg_rle in sem_masks.items():    
                    if class_id == '255':
                        continue    # ignore 'void' class
                    
                    if self.mask_area(sem_seg_rle, img_dims) >= MIN_MASK_AREA:
                        # lowest class_id could be 0
                        track_id = int(class_id) + 1
                        updated_segmentations[-1][track_id] = sem_seg_rle
                        accepted_track_ids[track_id] = int(class_id)

            # NOTE: the semantic classes in KITTI_STEP have IDs from 0-18 (and void/255).
            # The instance segmentations have their independent IDs that overlap with the
            # class IDs. To resolve this issue, the instances are assigned a new id as 
            # follows: new_id = max_track_id + 1 + real_id where max_track_id is the ID
            # of the highest class ID
            
            max_track_id = max(accepted_track_ids.keys())

            # store the IDs of the salient classes which have some of their instances 
            # segmented. This is used later to create a hole in the semantic map where 
            # instance-level masks are available
            salient_classes = []
            
            # read instance masks
            for fr_idx, inst_masks in enumerate(seq["segmentations"]):
                salient_classes.append(defaultdict(list))

                for track_id, inst_rle in inst_masks.items():

                    if self.mask_area(inst_rle, img_dims) >= MIN_MASK_AREA:
                        # new track ID
                        new_track_id = max_track_id + 1 + int(track_id)

                        updated_segmentations[fr_idx][new_track_id] = inst_rle
                        accepted_track_ids[new_track_id] = seq['categories'][track_id]
                        salient_classes[-1][int(seq['categories'][track_id]) + 1].append(self.decode_mask(inst_rle, img_dims))
            
            # cut out holes from the semantic map of the salient classes where instance masks are available
            for fr_idx, fr_rles in enumerate(updated_segmentations):
                overlapping_masks = salient_classes[fr_idx]
                if len(overlapping_masks) == 0:
                    continue
                
                for class_id in overlapping_masks.keys():
                    sem_mask = self.decode_mask(fr_rles[class_id], img_dims)
                    for inst_mask in overlapping_masks[class_id]:
                        sem_mask[np.where(inst_mask==1)] = 0
                
                    if np.any(sem_mask):
                        updated_sem_mask = mt.encode(np.asfortranarray(sem_mask))["counts"].decode('utf-8')
                        updated_segmentations[fr_idx][class_id] = updated_sem_mask
                    else:
                        updated_segmentations[fr_idx].pop(class_id, None)

            seq['segmentations'] = updated_segmentations
            seq["categories"] = accepted_track_ids
            
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

    
    
    def map_annotations_2(
            self, 
            annotations_path: str
    ):
        """
        Read KITTI-STEP annotations from JSON file

        Args:
            path_to_annotations: path to JSON annotations

        Returns a dictionary with annotation content from the entire dataset
        """
        
        # read JSON file
        with open(annotations_path, 'r') as fh:
            content = json.load(fh)

        sequences = []

        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA # no filtering applied

        for seq in content["sequences"]:
            
            seq['id'] = f"{self.name}/{seq['id']}"
            seq['dataset'] = self.name
            
            # video resolution
            img_dims = (seq['height'], seq['width'])
            
            updated_segmentations = []
            accepted_track_ids = {}

            # read semantic maps
            for fr_idx, sem_masks in enumerate(seq["semantic_segmentations"]):
                updated_segmentations.append(dict())

                for class_id, sem_seg_rle in sem_masks.items():    
                    if class_id == '255':
                        continue    # ignore 'void' class
                    
                    if self.mask_area(sem_seg_rle, img_dims) >= MIN_MASK_AREA:
                        # lowest class_id could be 0
                        track_id = int(class_id) + 1
                        updated_segmentations[-1][track_id] = sem_seg_rle
                        accepted_track_ids[track_id] = int(class_id)

            # NOTE: the semantic classes in KITTI_STEP have IDs from 0-18 (and void/255).
            # The instance segmentations have their independent IDs that overlap with the
            # class IDs. To resolve this issue, the instances are assigned a new id as 
            # follows: new_id = max_track_id + 1 + real_id where max_track_id is the ID
            # of the highest class ID
            
            max_track_id = max(accepted_track_ids.keys())

            # store the IDs of the salient classes which have some of their instances 
            # segmented. This is used later to create a hole in the semantic map where 
            # instance-level masks are available
            salient_classes = []
            
            # read instance masks
            for fr_idx, inst_masks in enumerate(seq["segmentations"]):
                salient_classes.append(defaultdict(list))

                for track_id, inst_rle in inst_masks.items():

                    if self.mask_area(inst_rle, img_dims) >= MIN_MASK_AREA:
                        # new track ID
                        new_track_id = max_track_id + 1 + int(track_id)

                        updated_segmentations[fr_idx][new_track_id] = inst_rle
                        accepted_track_ids[new_track_id] = seq['categories'][track_id]
                        salient_classes[-1][int(seq['categories'][track_id]) + 1].append(self.decode_mask(inst_rle, img_dims))
            
            # cut out holes from the semantic map of the salient classes where instance masks are available
            for fr_idx, fr_rles in enumerate(updated_segmentations):
                overlapping_masks = salient_classes[fr_idx]
                if len(overlapping_masks) == 0:
                    continue
                
                for class_id in overlapping_masks.keys():
                    sem_mask = self.decode_mask(fr_rles[class_id], img_dims)
                    for inst_mask in overlapping_masks[class_id]:
                        sem_mask[np.where(inst_mask==1)] = 0
                
                    if np.any(sem_mask):
                        updated_sem_mask = mt.encode(np.asfortranarray(sem_mask))["counts"].decode('utf-8')
                        updated_segmentations[fr_idx][class_id] = updated_sem_mask
                    else:
                        updated_segmentations[fr_idx].pop(class_id, None)

            seq['segmentations'] = updated_segmentations
            seq["categories"] = accepted_track_ids
            
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
        if img_dims is not None:
            encoded_mask = {
                "counts": rle.encode("utf-8"),
                "size": img_dims
            }
        else:
            encoded_mask = rle
        return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)


########################### INFERENCE DATASET ###########################


class KITTISTEPInferenceDataset(InferenceDataset):
    """
    Inference dataset for KITTI_STEP

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


    def create_inference_dataset(self, single_instance=False):
        """
        Read instance and semantic annotations from disc and prepare panoptic 
        segmentation masks.
        """

        sequences = []
        
        for seq in self.annotations["sequences"]:

            metadata = {"id": seq["id"]}
            img_dims = (seq['height'], seq['width'])

            # load images
            image_filepaths = sorted([os.path.join(self.path_to_images, seq['id'], file) for file in os.listdir(os.path.join(self.path_to_images, seq['id'])) if file.endswith('png')])
            images = self.load_images(image_filepaths)      # [T, H, W, 3]

            metadata["length"] = len(image_filepaths)
            metadata["orig_dims"] = (images.shape[1], images.shape[2])

            updated_segmentations = []
            accepted_track_ids = {}

            # read semantic maps
            for fr_idx, sem_masks in enumerate(seq["semantic_segmentations"]):
                
                updated_segmentations.append(dict())

                for class_id, sem_seg_rle in sem_masks.items():    
                    if class_id == '255':
                        continue    # ignore 'void' class
                    
                    # lowest class_id could be 0
                    track_id = int(class_id) + 1
                    updated_segmentations[-1][track_id] = sem_seg_rle
                    accepted_track_ids[track_id] = int(class_id)

            # NOTE: the semantic classes in KITTI_STEP have IDs from 0-18 (and void/255).
            # The instance segmentations have their independent IDs that overlap with the
            # class IDs. To resolve this issue, the instances are assigned a new id as 
            # follows: new_id = max_track_id + 1 + real_id where max_track_id is the ID
            # of the highest class ID
            
            max_track_id = max(accepted_track_ids.keys())

            # store the IDs of the salient classes which have some of their instances 
            # segmented. This is used later to create a hole in the semantic map where 
            # instance-level masks are available
            salient_classes = []
            
            # read instance masks
            for fr_idx, inst_masks in enumerate(seq["segmentations"]):
                
                salient_classes.append(defaultdict(list))
                for track_id, inst_rle in inst_masks.items():

                    # new track ID
                    new_track_id = max_track_id + 1 + int(track_id)

                    updated_segmentations[fr_idx][new_track_id] = inst_rle
                    accepted_track_ids[new_track_id] = seq['categories'][track_id]
                    salient_classes[-1][int(seq['categories'][track_id]) + 1].append(self.decode_mask(inst_rle, img_dims))
            
            # cut out holes from the semantic map of the salient classes where instance masks are available
            for fr_idx, fr_rles in enumerate(updated_segmentations):
                overlapping_masks = salient_classes[fr_idx]
                if len(overlapping_masks) == 0:
                    continue
                
                for class_id in overlapping_masks.keys():
                    sem_mask = self.decode_mask(fr_rles[class_id], img_dims)
                    for inst_mask in overlapping_masks[class_id]:
                        sem_mask[np.where(inst_mask==1)] = 0
                
                    updated_segmentations[fr_idx][class_id] = mt.encode(np.asfortranarray(sem_mask))

            seq['segmentations'] = updated_segmentations
            seq["categories"] = accepted_track_ids
            
            seq.pop("semantic_segmentations")
            sequences.append(seq)

        # store category id to name mapping
        meta_info = self.annotations["meta"]["category_labels"]
        meta_info = {
            "category_labels": {
                int(id): name for id, name in self.annotations["meta"]["category_labels"].items()
            }
        }

        return {
            "sequences": sequences,
            "meta": meta_info
        }
    
    # def create_inference_dataset(self, single_instance=False):
    #     """
    #     Prepare dataset for evaluation.
    #     """

    #     if self.split != "val":
    #         raise RuntimeError(f"Annotations for KITTI-STEP {self.split} split is not available")
        
    #     sequence_annotations = []
        
    #     for seq in self.annotations["sequences"]:

    #         metadata = {"id": seq['id']}

    #         # for each instance (class), store index of the frame where it first appeared
    #         instance_discovery = {}

    #         # generate panoptic masks
    #         updated_segmentations = []      # store valid segmentations (binary)
    #         salient_classes = []            # store salient class IDs whose instances are present in the frame
    #         accepted_track_ids = {}         # store accepted instance IDs
            
    #         # salient classes ('person' and 'car') - with instance-level annotations
    #         for fr_idx, segs_t in enumerate(seq['segmentations']):
                
    #             updated_segmentations.append(dict())
    #             salient_classes.append(set())
                
    #             # add instance masks of the salient classes
    #             for track_id, seg in segs_t.items():
    #                 # store instance mask (decoded from RLE)
    #                 updated_segmentations[-1][int(track_id)] = self.decode_mask(seg, metadata["orig_dims"])
    #                 accepted_track_ids[int(track_id)] = seq['categories'][track_id]
    #                 # note the salient classes present in the frame
    #                 salient_classes[-1].add(seq['categories'][track_id])
                    
    #                 if int(track_id) not in instance_discovery.keys():
    #                     instance_discovery[int(track_id)] = fr_idx

    #         # maximum instance ID (belonging only to salient classes) seen across all frames
    #         max_track_id = max(accepted_track_ids.keys())
            
    #         # semantic masks
    #         # Label values for panoptic class annotations start from max_track_id + 1 
    #         for fr_idx, pano_masks in enumerate(seq["semantic_segmentations"]):
    #             for class_id, pano_seg in pano_masks.items():
                    
    #                 # ignore if 'void' class
    #                 if class_id == '255':
    #                     continue
                    
    #                 # skip any annotation that belongs to the salient instances present in this frame
    #                 if int(class_id) not in salient_classes[fr_idx]:
    #                     # store panoptic mask with unique ID per 'stuff' class
    #                     stuff_track_id = max_track_id + int(class_id) + 1
    #                     updated_segmentations[fr_idx][stuff_track_id] = self.decode_mask(pano_seg, metadata["orig_dims"])
    #                     accepted_track_ids[int(stuff_track_id)] = int(class_id)

    #                     if int(stuff_track_id) not in instance_discovery.keys():
    #                         instance_discovery[int(stuff_track_id)] = fr_idx
            
    #         # find all the instances (classes) present in the sequence, so that we can insert empty
    #         # masks for the ones that are absent
    #         orig_instance_ids = sorted(list(instance_discovery.keys()))
    #         orig_to_serial_ids, serial_to_orig_ids = self.serialize_instance_ids(orig_instance_ids)
    #         assert orig_instance_ids == list(orig_to_serial_ids.keys())
    #         metadata["orig_to_serial_ids"] = orig_to_serial_ids
    #         metadata["serial_to_orig_ids"] = serial_to_orig_ids
            
    #         instance_discovery = {orig_to_serial_ids[inst_id]: fr_idx 
    #                                 for inst_id, fr_idx in instance_discovery.items()}
    #         instance_discovery = dict(sorted(instance_discovery.items()))
    #         metadata["instance_discovery"] = instance_discovery

    #         instance_ids = sorted(instance_discovery.keys())
            
    #         # load panoptic masks
    #         instance_masks = []
    #         semantic_maps = []
    #         instances_per_frame = []
            
    #         for fr_idx, pano_masks in enumerate(updated_segmentations):

    #             fr_instance_masks = []
    #             fr_semantic_map = np.zeros(metadata["orig_dims"])
    #             fr_instance_ids = []
                
    #             for inst_id in instance_ids:
    #                 if serial_to_orig_ids[inst_id] in pano_masks.keys():
    #                     fr_instance_ids.append(inst_id)
                        
    #                     msk = pano_masks[serial_to_orig_ids[inst_id]]
    #                     fr_semantic_map[np.where(msk==1)] = serial_to_orig_ids[inst_id]
    #                     fr_instance_masks.append(msk.astype(np.uint8))

    #                 else:
    #                     fr_instance_masks.append(np.zeros(metadata["orig_dims"]).astype(np.uint8))
                    
    #             instance_masks.append(np.stack(fr_instance_masks))
    #             semantic_maps.append(fr_semantic_map.astype(np.uint8))
    #             instances_per_frame.append(fr_instance_ids)
            
    #         instance_masks = np.stack(instance_masks)               # T, N, H, W
    #         semantic_maps = np.stack(semantic_maps)                 # T, H, W

    #         if self.cfg.INPUT.AUGMENTATION.RESIZE_TEST:
    #             # compute target resolution
    #             new_height, new_width = compute_resized_dims(
    #                 *images.shape[1:3], 
    #                 min_dim=self.cfg.INPUT.AUGMENTATION.MIN_DIM_TEST,
    #                 max_dim=self.cfg.INPUT.AUGMENTATION.MAX_DIM_TEST,
    #             )
    #             if (new_height, new_width) != metadata["orig_dims"]:
    #                 images = resize_images(images, new_height, new_width)
    #                 semantic_maps = resize_masks(semantic_maps, new_height, new_width, binary=False)
    #                 instance_masks = resize_masks(instance_masks, new_height, new_width, binary=True)
            
    #         # arrange dimensions
    #         images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
    #         if self.cfg.INPUT.RGB:
    #             # BGR -> RGB (load_images uses cv2.imread which reads images in BGR mode by default)
    #             images = np.flip(images, 1).copy()
            
    #         metadata["images"] = images
    #         metadata["bg_masks"] = (semantic_maps==0).astype(np.uint8)
    #         metadata["semantic_maps"] = semantic_maps
    #         metadata["instance_masks"] = instance_masks

    #         # TODO - padding - not applied
    #         metadata["padding_mask"] = np.zeros((images.shape[2], images.shape[3])).astype(np.uint8)

    #         metadata["instances_per_frame"] = instances_per_frame
    #         metadata["clip_length"] = self.clip_length
    #         metadata["num_overlapping_frames"] = self.num_overlapping_frames
            
    #         sequence_annotations.append(metadata)
        
    #     return sequence_annotations