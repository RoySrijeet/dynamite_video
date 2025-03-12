import os
import json
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

    Creates a `torch.utils.data.Dataset` class to load DAVIS dataset
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
        # annotations_content = self.map_annotations_IO(path_to_images, Paths.to_davis_annotations(), Paths.to_davis_train_imset())
        annotations_content = self.map_annotations(path_to_images, Paths.to_davis_train_annotations_json())
        
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
            path_to_images: str,
            path_to_annotations: str, 
    ):
        """
        Read semantic masks from PNG files

        Args:
            path_to_images: path to image directory
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
            entry["image_paths"] = [os.path.join(path_to_images, fpath) for fpath in seq["image_paths"]]

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

            # TODO - remove
            if len(sequences)==2:
                break

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
    
    
    # def map_annotations_IO(
    #         self,
    #         path_to_images: str,
    #         path_to_annotations: str, 
    #         path_to_imset: str, 
    # ):
    #     """
    #     Read semantic masks from PNG files

    #     Args:
    #         path_to_images: path to image directory
    #         path_to_annotations: path to annotations directory
    #         path_to_imset: path to imset .txt file, listing training sequence names

    #     Returns a dictionary with annotation content from the entire dataset
    #     """
        
    #     # load the list of training sequences as a list
    #     with open(path_to_imset, 'r') as f:
    #         sequences = [seq.rstrip() for seq in f.readlines()]

    #     sequence_annotations = []

    #     MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        
    #     # for each video sequence in the dataset
    #     for seq in sequences:
    #         entry = {}
    #         entry["id"] = f"{self.name}/{seq}"
    #         entry["dataset"] = self.name

    #         # load paths to all image files of the sequence
    #         imagefiles = sorted([os.path.join(path_to_images, seq, file) for file in os.listdir(os.path.join(path_to_images, seq)) if file.endswith('jpg')])
    #         entry["image_paths"] = imagefiles
            
    #         # load paths to all mask files of a sequence
    #         maskfiles = sorted([os.path.join(path_to_annotations, seq, file) for file in os.listdir(os.path.join(path_to_annotations, seq)) if file.endswith('png')])

    #         # read first frame mask again to store resolution and instances
    #         mask0 = np.asarray(Image.open(maskfiles[0]))
    #         entry["height"], entry["width"] = mask0.shape

    #         segmentations = []
    #         seq_instances = []
    #         # for mask of each frame of the video sequence
    #         for idx, file in enumerate(maskfiles):
    #             # read and store the semantic map
    #             mask = np.asarray(Image.open(file).convert("P")).astype(dtype='uint8')
    #             # find how many instances in the mask (excluding bg, value 0)
    #             instances = list(np.unique(mask))[1:]

    #             # extract and store binary masks of individual instances from semantic mask
    #             binary_masks = {}
    #             for i in instances:
    #                 _m = (mask==i).astype(dtype='uint8')
    #                 # check mask area
    #                 if self.mask_area(_m) >= MIN_MASK_AREA:
    #                     # if mask larger than threshold, keep it
    #                     binary_masks[int(i)] = mt.encode(np.asfortranarray(_m))
    #                     seq_instances.append(i)
    #             segmentations.append(binary_masks)

    #         seq_instances = set(seq_instances)
    #         entry["categories"] = {int(k):1 for k in seq_instances}
    #         entry['segmentations'] = segmentations

    #         sequence_annotations.append(entry)

    #         # TODO - remove
    #         if len(sequences)==2:
    #             break

    #     annotations_content = {}
    #     # there is no explicit categories present in DAVIS
    #     annotations_content["meta"] = {"category_labels": {1: 'object'}}
    #     annotations_content["sequences"] = sequence_annotations

    #     return annotations_content

    # def mask_area_IO(self, mask):
    #     assert isinstance(mask, np.ndarray)
    #     bin_mask = mask.astype('uint8')
    #     assert list(np.unique(bin_mask))==[0,1]
    #     return bin_mask.sum()


class DAVISInferenceDataset(InferenceDataset):
    
    def __init__(self, cfg):
        # number of frames in each training sample
        clip_length = cfg.DATASETS.DAVIS.INFERENCE.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.DAVIS.INFERENCE.FPS
        # number of overlapping frames between clips
        num_overlapping_frames = cfg.DATASETS.DAVIS.INFERENCE.FRAME_OVERLAP
        split = cfg.DATASETS.DAVIS.INFERENCE.SPLIT
        # DAVIS has only one "val" split
        assert split=="val"
        
        super().__init__(cfg, "DAVIS", clip_length, fps, num_overlapping_frames, split)

        # get paths
        path_to_images = Paths.to_davis_images()
        # `images_dir` could be an fpack file
        if not os.path.exists(path_to_images):
            path_to_images = f"{path_to_images}.fpack"
            assert os.path.exists(path_to_images), f"Directory not found: {path_to_images}"
        

        # load masks of video frames in DAVIS training split 
        annotations_content = self.map_annotations(path_to_images, Paths.to_davis_annotations(), Paths.to_davis_val_imset())

        self.annotation_clips = self.create_inference_clips(annotations_content)


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
        
        # load the list of evaluation sequences as a list
        with open(path_to_imset, 'r') as f:
            sequences = [seq.rstrip() for seq in f.readlines()]

        sequence_annotations = []
        
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
                    binary_masks[int(i)] = mt.encode(np.asfortranarray(_m))['counts'].decode("utf-8")
                    seq_instances.append(i)
                segmentations.append(binary_masks)

            seq_instances = set(seq_instances)
            entry["categories"] = {int(k):1 for k in seq_instances}
            entry['segmentations'] = segmentations

            sequence_annotations.append(entry)

            # TODO - remove
            if len(sequences)==2:
                break

        annotations_content = {}
        # there is no explicit categories present in DAVIS
        annotations_content["meta"] = {"category_labels": {1: 'object'}}
        annotations_content["sequences"] = sequence_annotations

        return annotations_content


