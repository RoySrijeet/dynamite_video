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

class DAVISTrainingDataset(TrainingDataset):
    """DAVIS Training Dataset Class"""
    
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
        self.path_to_images = Paths.to_davis_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/davis.fpack"
            assert os.path.exists(path_to_images), f"DAVIS images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)
        
        # load masks of video frames in DAVIS training split
        annotations_content = self.map_annotations(Paths.to_davis_train_annotations_json())

        # load masks from PNG files and store them as RLEs
        # annotations_content = self.map_annotations_IO(Paths.to_davis_annotations(), Paths.to_davis_train_imset())
        
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
        
        # load the list of training sequences as a list
        with open(path_to_annotations, 'r') as f:
            content = json.load(f)

        sequences = []

        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
        # for each video sequence in the dataset
        for seq in content["sequences"]:
            
            entry = {}
            entry["id"] = f"{self.name}/{seq['id']}"
            entry["dataset"] = self.name
            entry["height"] = seq["height"]
            entry["width"] = seq["width"]
            img_dims = (seq['height'], seq['width'])

            # relative paths to all JPEG images of the dataset (only the annotated ones)
            entry["image_paths"] = seq["image_paths"]

            updated_segmentations = []
            accepted_track_ids = set()

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

            entry['segmentations'] = updated_segmentations
            entry["categories"] = {
                int(track_id): seq["categories"][track_id]
                for track_id in accepted_track_ids
            }
            
            sequences.append(entry)

        annotations_content = {}
        # there is no explicit categories present in DAVIS
        annotations_content["meta"] = {"category_labels": {1: 'object'}}
        annotations_content["sequences"] = sequences
        del content

        return annotations_content
    
    def mask_area(self, rle, img_dims):
        """
        Area of an RLE segment
        """
        return mt.area({
            "counts": rle.encode("utf-8"),
            "size": img_dims
        })
    

    def map_annotations_IO(
            self,
            path_to_annotations: str, 
            path_to_imset: str, 
    ):
        """
        Read semantic masks from PNG files.

        NOTE: Considerably slower as involves reading PNG files and converting them to RLEs

        Args:
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
            imagefiles = sorted([os.path.join(self.path_to_images, seq, file) for file in os.listdir(os.path.join(self.path_to_images, seq)) if file.endswith('jpg')])
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
                    if self.mask_area_array(_m) >= MIN_MASK_AREA:
                        # if mask larger than threshold, keep it
                        binary_masks[int(i)] = mt.encode(np.asfortranarray(_m))
                        seq_instances.append(i)
                segmentations.append(binary_masks)

            seq_instances = set(seq_instances)
            entry["categories"] = {int(k):1 for k in seq_instances}
            entry['segmentations'] = segmentations

            sequence_annotations.append(entry)

        annotations_content = {}
        # there is no explicit categories present in DAVIS
        annotations_content["meta"] = {"category_labels": {1: 'object'}}
        annotations_content["sequences"] = sequence_annotations

        return annotations_content


    def mask_area_array(self, mask):
        assert isinstance(mask, np.ndarray)
        bin_mask = mask.astype('uint8')
        assert list(np.unique(bin_mask))==[0,1]
        return bin_mask.sum()



########################### INFERENCE DATASET ###########################


class DAVISInferenceDataset(InferenceDataset):
    """
    Inference dataset for DAVIS ("val" split).

    Loads image and mask files from the disc and generates indices
    of clips that are to be used in inference forward pass.
    """
    
    def __init__(self, cfg):
        # number of frames in each training sample
        clip_length = cfg.DATASETS.DAVIS.INFERENCE.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.DAVIS.INFERENCE.FPS
        # number of overlapping frames between clips
        num_overlapping_frames = cfg.DATASETS.DAVIS.INFERENCE.FRAME_OVERLAP

        assert num_overlapping_frames <= clip_length, f"No. of overlapping frames cannot be more than the length of a clip"
        
        split = cfg.DATASETS.DAVIS.INFERENCE.SPLIT
        # DAVIS has only one "val" split
        assert split=="val"
        
        super().__init__(cfg, "DAVIS", clip_length, fps, num_overlapping_frames, split)

        # get paths
        self.path_to_images = Paths.to_davis_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_evaluation_images_on_juwels()}/davis.fpack"
            assert os.path.exists(path_to_images), f"DAVIS images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)

        self.path_to_annotations = Paths.to_davis_annotations()
        self.path_to_val_imset = Paths.to_davis_val_imset()

    
    def create_inference_dataset(self, single_instance=False):
        """
        Prepare dataset for evaluation. Involves the following steps:
            1. Load images and masks from files
            2. Serialize instance IDs. NOTE: All structures loaded in this routine use the 
            serialized IDs
            3. Generate indices for extracting clips/sub-sequences from each sequence
            that is to be used for inference forward pass
        
        For each sequence, return a dictionary containing the following keys:
            * "id": str, sequence name
            * "length": int, length of the sequence (T)
            * "orig_dims": tuple(int), original resolution of sequence frames
            * "images": [T,3,H,W] np.ndarray, RGB images of the sequence frames
            * "instance_masks": [T,N,H,W] np.ndarray, binary segmentation masks of 
                        instances in each frame (N: #instances in the sequence)
            * "semantic_maps": [T,H,W] np.ndarray, semantic map of each frame
            * "bg_masks": [T,H,W] np.ndarray, background mask of each frame
            * "instances_per_frame": list, IDs of instances present in each frame
            * "padding_mask": [H,W] np.ndarray, padding mask (a 0-array)
            * "orig_to_serial_ids": dict, mapping between original instance IDs and 
                        serialied instance IDs used by the model
            * "serial_to_orig_ids": dict, mapping between serial instance IDs used 
                        by the model and the original instance IDs
            * "instance_discovery": dict, mapping between each instance ID and the 
                        frame index where the instance first appeared
            * "indices": list, frame indices for creating clips/sub-sequences

        """

        if single_instance:
            return self.create_single_inst_inference_dataset()

        # load the list of evaluation sequences as a list
        with open(self.path_to_val_imset, 'r') as f:
            sequences = [seq.rstrip() for seq in f.readlines()]
        
        sequence_annotations = []
        for seq in sequences:
            
            metadata = {"id": seq}

            # load images
            image_filepaths = sorted([os.path.join(self.path_to_images, seq, file) for file in os.listdir(os.path.join(self.path_to_images, seq)) if file.endswith('jpg')])
            images = self.load_images(image_filepaths)      # [T, H, W, 3]

            metadata["length"] = len(image_filepaths)
            metadata["orig_dims"] = (images.shape[1], images.shape[2])

            # semantic_maps - [T,H,W] np.ndarray
            # bg_masks - [T,H,W] np.ndarray binary background masks
            # instances_per_frame - list of IDs of instances present in each frame
            # instance_discovery - dict, instance ID and frame index where the instance first appeared
            mask_filepaths = sorted([os.path.join(self.path_to_annotations, seq, file) for file in os.listdir(os.path.join(self.path_to_annotations, seq)) if file.endswith('png')])
            semantic_maps, bg_masks, instances_per_frame, instance_discovery = self.load_png_masks(mask_filepaths)
            
            # resize
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
                    bg_masks = resize_masks(bg_masks, new_height, new_width, binary=False)
            
            # arrange dimensions
            images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
            if self.cfg.INPUT.RGB:
                # BGR -> RGB (load_images uses cv2.imread which reads images in BGR mode by default)
                images = np.flip(images, 1).copy()
            
            metadata["images"] = images
            metadata["bg_masks"] = bg_masks
            # TODO - padding - not applied
            metadata["padding_mask"] = np.zeros((images.shape[2], images.shape[3])).astype('uint8')
            
            # serialize instance IDs
            
            # IDs of all instances present in the sequence
            orig_instance_ids = sorted(list(instance_discovery.keys()))
            orig_to_serial_ids, serial_to_orig_ids = self.serialize_instance_ids(orig_instance_ids)
            assert orig_instance_ids == list(orig_to_serial_ids.keys())
            metadata["orig_to_serial_ids"] = orig_to_serial_ids
            metadata["serial_to_orig_ids"] = serial_to_orig_ids

            metadata["instance_discovery"] = {orig_to_serial_ids[inst_id]: fr_idx 
                                              for inst_id, fr_idx in instance_discovery.items()}
            
            
            # extract binary instance masks from semantic map of each frame
            sequence_instances_serial_ids = sorted(list(metadata["instance_discovery"].keys()))
            instance_masks = []
            for fr_idx, fr_map in enumerate(semantic_maps):
                fr_inst_masks = []
                for inst_id in sequence_instances_serial_ids:
                    # semantic map is still labeled with original instance IDs
                    orig_inst_id = serial_to_orig_ids[inst_id]
                    if orig_inst_id in instances_per_frame[fr_idx]:
                        fr_inst_masks.append((fr_map == orig_inst_id).astype('uint8'))
                    else:
                        # if an instance is absent, pad it with empty mask
                        fr_inst_masks.append(np.zeros_like(fr_map).astype('uint8'))
                instance_masks.append(np.stack(fr_inst_masks))
            
            instance_masks = np.stack(instance_masks)    # [T,N,H,W]
            metadata["instance_masks"] = instance_masks

            # recreate semantic maps with serial instance IDs
            semantic_maps_serial = []
            for fr_idx, fr_inst_masks in enumerate(instance_masks):
                fr_map = np.zeros_like(fr_inst_masks[0])
                for inst_id, inst_mask in zip(sequence_instances_serial_ids, fr_inst_masks):
                    fr_map[np.where(inst_mask==1)] = inst_id
                semantic_maps_serial.append(fr_map.astype('uint8'))
                
                # update record on which instances are present in this frame
                instances_per_frame[fr_idx] = [orig_to_serial_ids[orig_inst_id] for orig_inst_id in instances_per_frame[fr_idx]]
            
            metadata["semantic_maps"] = np.stack(semantic_maps_serial)
            metadata["instances_per_frame"] = instances_per_frame

            metadata["clip_length"] = self.clip_length
            metadata["num_overlapping_frames"] = self.num_overlapping_frames

            sequence_annotations.append(metadata)

        return sequence_annotations


    def create_single_inst_inference_dataset(self):
        """
        Create single instance inference dataset. Sequences containing more than one 
        instances are separated into multiple sequences with one instance each
        """
        
        # load the list of evaluation sequences as a list
        with open(self.path_to_val_imset, 'r') as f:
            sequences = [seq.rstrip() for seq in f.readlines()]
        
        sequence_annotations = []
        for seq in sequences:
            
            # load images
            image_filepaths = sorted([os.path.join(self.path_to_images, seq, file) for file in os.listdir(os.path.join(self.path_to_images, seq)) if file.endswith('jpg')])
            images = self.load_images(image_filepaths)      # [T, H, W, 3]
            orig_dims = (images.shape[1], images.shape[2])

            # load masks
            mask_filepaths = sorted([os.path.join(self.path_to_annotations, seq, file) for file in os.listdir(os.path.join(self.path_to_annotations, seq)) if file.endswith('png')])
            multi_semantic_maps = []
            for fr_idx, fp in enumerate(mask_filepaths):
                msk = np.asarray(Image.open(fp)).astype('uint8')
                multi_semantic_maps.append(msk)
            multi_semantic_maps = np.stack(multi_semantic_maps)

            # resize
            if self.cfg.INPUT.AUGMENTATION.RESIZE_TEST:
                # compute target resolution
                new_height, new_width = compute_resized_dims(
                    *images.shape[1:3], 
                    min_dim=self.cfg.INPUT.AUGMENTATION.MIN_DIM_TEST,
                    max_dim=self.cfg.INPUT.AUGMENTATION.MAX_DIM_TEST,
                )
                if (new_height, new_width) != orig_dims:
                    images = resize_images(images, new_height, new_width)
                    multi_semantic_maps = resize_masks(multi_semantic_maps, new_height, new_width, binary=False)

            # arrange dimensions
            images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
            if self.cfg.INPUT.RGB:
                # BGR -> RGB (load_images uses cv2.imread which reads images in BGR mode by default)
                images = np.flip(images, 1).copy()

            padding_mask = np.zeros((images.shape[2], images.shape[3])).astype('uint8')

            # In DAVIS, all instances are assumed to be present in the first frame
            all_instance_ids = list(np.unique(multi_semantic_maps[0]))[1:]
            for inst_id in all_instance_ids:
                
                metadata = {"id": f"{seq}_{inst_id}"}
                metadata["images"] = images
                metadata["length"] = len(image_filepaths)
                metadata["orig_dims"] = orig_dims

                metadata["orig_to_serial_ids"] = {1: 1}
                metadata["serial_to_orig_ids"] = {1: 1}
                metadata["instance_discovery"] = {1: 0}
                
                # semantic_maps - [T,H,W] np.ndarray
                semantic_maps = (multi_semantic_maps == inst_id).astype('uint8')
                
                metadata["instance_masks"] = np.expand_dims(semantic_maps, axis=1)  # [T,N,H,W]
                metadata["bg_masks"] = (semantic_maps==0).astype('uint8')           # [T,H,W]
                metadata["padding_mask"] = padding_mask                             # [H,W]
                metadata["semantic_maps"] = semantic_maps                           # [T,H,W]
                
                # instances_per_frame - list of IDs of instances present in each frame
                metadata["instances_per_frame"] = [list(np.unique(msk))[1:] for msk in semantic_maps]
                

                metadata["clip_length"] = self.clip_length
                metadata["num_overlapping_frames"] = self.num_overlapping_frames

                sequence_annotations.append(metadata)

        return sequence_annotations