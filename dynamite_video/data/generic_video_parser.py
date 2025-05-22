# Adapted from https://github.com/Ali2500/TarViS

import os
import cv2
import json
import numpy as np
import pycocotools.mask as mt

from collections import OrderedDict
from typing import Any, List, Dict, Tuple, Optional, Union


from dynamite_video.data.utils.file_packer import FilePackReader


def parse_generic_video_dataset(
        path_to_images: str, 
        annotations_content: Union[str, Dict[str, Any]]
):
    """
    Cast each sequence in the dataset to `GenericVideoSequence` class template

    Args:
        path_to_images: path to dataset images
        annotations_content: dataset annotation, may be a path to a JSON file
            or a dict with mask information
    """

    # load annotations from the JSON file
    if isinstance(annotations_content, str):
        with open(annotations_content, 'r') as fh:
            dataset = json.load(fh)
    else:
        # if the annotation dict is already passed as an argument
        dataset = annotations_content

    meta_info = dataset["meta"]

    if "category_labels" in meta_info:
        # sanity check: instance IDs in "segmentations" must match those in "categories"
        for seq in dataset["sequences"]:
            if "categories" in seq:
                
                seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
                assert seg_iids == set(seq["categories"].keys()), "Instance ID mismatch in seq {}: {} vs. {}".format(
                    seq["id"], seg_iids, set(seq["categories"].keys())
                )

    # wrap each video sequence as a `GenericVideoSequence` object
    seqs = [GenericVideoSequence(seq, path_to_images) for seq in dataset["sequences"]]

    return seqs, meta_info


class GenericVideoSequence(object):
    """
    Generic Video Sequence template
    
    A common format for video sequences from different datasets as well as
    sub-sequences generated in runtime
    """
    
    def __init__(
            self, 
            seq_dict,
            path_to_images,
            serialize=False
    ):
        """
        Args:
            seq_dict: dictionary containing info (resolution, annotation) of a 
                single video sequence from a dataset
            path_to_images: path to dataset images
            serialize: serialize non-sequential instance ids
        """
        assert len(seq_dict["image_paths"]) == len(seq_dict["segmentations"])

        self.path_to_images = path_to_images
        self.image_paths = seq_dict["image_paths"]
        self.image_dims = (seq_dict["height"], seq_dict["width"])
        self.id = seq_dict["id"]

        if not serialize:
            self.segmentations = seq_dict["segmentations"]
            self.semantic_segmentations = seq_dict.get("semantic_segmentations", None)
            self.instance_categories = seq_dict["categories"]
            inst_ids = sorted(seq_dict["categories"].keys())
            self.orig_to_serial_id = OrderedDict(zip(inst_ids, inst_ids))
            self.serial_to_orig_id = OrderedDict(zip(inst_ids, inst_ids))

        # serialize non-sequential instance IDs
        else:
            # obtain mappings between original and sequential IDs
            self.orig_to_serial_id, self.serial_to_orig_id = self.serialize_instance_ids(sorted(seq_dict["categories"].keys()))
            if all(key == value for key, value in self.orig_to_serial_id.items()):
                # already serial
                self.segmentations = seq_dict["segmentations"]
                self.semantic_segmentations = seq_dict.get("semantic_segmentations", None)
                self.instance_categories = seq_dict["categories"]
            else:
                # serialize seg keys and labels
                segmentations = seq_dict["segmentations"]
                semantic_segmentations = seq_dict.get("semantic_segmentations", None)
                self.segmentations, self.semantic_segmentations = self.serialize_masks(segmentations, semantic_segmentations, self.orig_to_serial_id)
                # update IDs in category map
                self.instance_categories = {self.orig_to_serial_id.get(inst_id): value for inst_id, value in seq_dict["categories"].items()}

        self.ignore_masks = seq_dict.get("ignore_masks", None)
        self.instance_areas = None
        self.fpack_reader = None


    def serialize_instance_ids(self, orig_ids):
        """
        Serialize instance IDs. IDs are 1-indexed to avoid conflict in semantic mask
        with background pixels (0)

        Args:
            orig_ids: original instance IDs, potentially non-sequential

        Returns:
            orig_to_serial_id: mapping from original IDs to sequential IDs
            serial_to_orig_id: mapping from sequential IDs to original IDs
        """
        
        orig_ids = sorted(orig_ids)
        serial_ids = [i for i in range(1, len(orig_ids)+1)]
        serial_to_orig_id = OrderedDict(zip(serial_ids, orig_ids))
        orig_to_serial_id = OrderedDict(zip(orig_ids, serial_ids))
        return orig_to_serial_id, serial_to_orig_id
    
    
    def serialize_masks(self, segmentations, semantic_segmentations, orig_to_serial_id):
        """
        Rename annotation keys from original instance IDs to sequential IDs

        Args:
            segmentations: dictionary of instance masks where original instance IDs
                were used as keys
            semantic_segmentation: dictionary of semantic masks where original
                instance IDs are used as pixel labels
            orig_to_serial_ids: mapping from original IDs to sequential IDs
        """
        
        # update instance masks
        updated_segmentations = []
        for fr_idx, segs_t in enumerate(segmentations):
            # update keys
            new_seg = {orig_to_serial_id.get(inst_id): value for inst_id, value in segs_t.items()}
            updated_segmentations.append(new_seg)
            
        updated_semantic_segmentations = None
        # update semantic masks
        if semantic_segmentations is not None:
            updated_semantic_segmentations = {}
            for fr_idx, map in semantic_segmentations.items():
                new_map = np.zeros_like(map)
                for orig_id, serial_id in orig_to_serial_id.items():
                    # update pixel labels
                    new_map[np.where(map==orig_id)] = serial_id
                updated_semantic_segmentations[fr_idx] = new_map.astype('uint8')
            
        return updated_segmentations, updated_semantic_segmentations

    
    @property
    def height(self):
        """Height of each video frame"""
        return self.image_dims[0]


    @property
    def width(self):
        """Width of each video frame"""
        return self.image_dims[1]


    @property
    def instance_ids(self):
        """IDs of instances present in the video"""
        return list(self.instance_categories.keys())


    @property
    def category_labels(self):
        """Category labels of the instances in the video"""
        return {inst_id: self.instance_categories[inst_id] for inst_id in self.instance_ids}


    @property
    def has_semantic_masks(self):
        """Whether semantic maps are present or not"""
        return self.semantic_segmentations is not None
    

    def __len__(self):
        """Number of frames in the video"""
        return len(self.image_paths)


    def load_images(self, frame_idxes=None):
        """
        Load frames from the video sequence from disc.

        NOTE: the images are loaded in `cv2.imread(f, flags=cv2.IMREAD_COLOR)` mode,
        which is the default behavior. This loads the image in BGR format, not RGB.

        Args:
            frame_idxes: if set, load JPGs of specified frames

        Returns:
            images: a list of `np.ndarray`, each array being an [H, W, 3] image (uint8)
        """
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        if self.path_to_images.endswith(".fpack") and self.fpack_reader is None:
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)

        images = []
        for t in frame_idxes:
            if self.fpack_reader is None:
                im = cv2.imread(os.path.join(self.path_to_images, self.image_paths[t]), cv2.IMREAD_COLOR)
            else:
                im = self.fpack_reader.cv2_imread(self.image_paths[t], cv2.IMREAD_COLOR, exclude_base_path=True)

            if im is None:
                raise ValueError("No image found at path: {}".format(os.path.join(self.path_to_images, self.image_paths[t])))
            images.append(im)

        images = np.stack(images)       # [T, H, W, 3]
        return images

    
    def decode_mask(self, encoded_mask: Union[str, List[int]], size=None):
        """
        Decode RLE mask into `np.ndarray`

        Args:
            encoded_mask: RLE mask
            size: mask dimensions
        
        Returns:
            `np.ndarray` of dimensions `size`
        """
        if size is None:
            assert isinstance(encoded_mask, dict)
            assert 'counts' in encoded_mask.keys()
            assert 'size' in encoded_mask.keys()
            return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)

        if isinstance(encoded_mask, list):  # polygons
            encoded_mask = {
                "counts": encoded_mask,
                "size": size,
            }
            encoded_mask = mt.frPyObjects(encoded_mask, size[0], size[1])
        
        else:  # RLE mask
            assert isinstance(encoded_mask, str), f"Unexpected encoded mask type: {type(encoded_mask)}"
            encoded_mask = {
                "counts": encoded_mask.encode("utf-8"),
                "size": size
            }
        
        return np.ascontiguousarray(mt.decode(encoded_mask)).astype(np.uint8)
    

    def prepare_masks(self):
        """
        Decode RLE masks to numpy arrays of segmentation maps

        Returns:
            binary_masks: binary segmentation masks of the instances in the frames of the video
                as a list of lists of `np.ndarray`
            num_instances_per_frame: record of #instances present in each frame separately, as there 
                may be instances with empty masks which do not count
            instance_ids: IDs of instances in the clip, arranged in the same order as the binary masks
        """
        clip_instance_ids = sorted(self.instance_ids)

        # binary masks of each frame in the clip is made to have same no. 
        # of channels as the total number of instances in the clip. So, 
        # if an instance is not present in a frame, add an empty mask
        binary_masks = []
        
        # IDs of instances present in each frame
        instances_per_frame = []
        
        for fr_idx, fr_masks in enumerate(self.segmentations):
            instances_per_frame.append(sorted(fr_masks.keys()))
            
            binary_masks_fr = []
            # add one channel for each instance present in the clip
            for inst_id in clip_instance_ids:
                if inst_id in instances_per_frame[-1]:
                    # decode RLE
                    rle = fr_masks[inst_id]
                    img_dims = None if isinstance(rle, dict) else self.image_dims
                    _m = self.decode_mask(rle, img_dims)
                    # record
                    binary_masks_fr.append(_m)
                else:
                    binary_masks_fr.append(np.zeros(self.image_dims).astype('uint8'))

            binary_masks.append(binary_masks_fr)
        
        binary_masks = np.stack([np.stack(fr_masks) for fr_masks in binary_masks])      # [T, N, H, W]
        
        # ignore masks
        ignore_masks = None
        if self.ignore_masks is not None:
            ignore_masks = [self.decode_mask(ig_msk, img_dims) for ig_msk in self.ignore_masks]
            ignore_masks = np.stack(ignore_masks)
        
        return binary_masks, instances_per_frame, clip_instance_ids, ignore_masks
            

    def extract_subsequence(self, frame_idxes: List[int], instance_ids_to_keep: List[int]=None, new_id: str=""):
        """
        Extract the specified frames from the video and return it as a clip

        The clip is also cast as a `GenericVideoSequence` object

        Args:
            frame_idxes: frames to be extracted from the main video
            instance_ids_to_keep: instances to keep
            new_id: new id for the subsequence

        Returns:
            A GenericVideoSequence object with the specified frames and instances
        """

        assert all([t in range(len(self)) for t in frame_idxes])

        subseq_dict = {
            "id": new_id if new_id else self.id,
            "height": self.image_dims[0],
            "width": self.image_dims[1],
            "image_paths": [self.image_paths[t] for t in frame_idxes]
        }
        
        if instance_ids_to_keep is None:
            
            subseq_dict["segmentations"] = []
            subseq_instances = []
            for fr_idx in frame_idxes:
                subseq_dict["segmentations"].append(self.segmentations[fr_idx])
                subseq_instances.extend(self.segmentations[fr_idx].keys())

            subseq_instances = set(subseq_instances)
            subseq_dict["categories"] = {iid: self.instance_categories[iid] for iid in subseq_instances}
        
        else:
            subseq_dict["segmentations"] = [
                {
                    iid: segmentations_t[iid]
                    for iid in segmentations_t if iid in instance_ids_to_keep
                }
                for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
            ]

            subseq_dict["categories"] = {iid: self.instance_categories[iid] for iid in instance_ids_to_keep}

            # if self.has_semantic_masks:
            #     subseq_semantic_segmentation = {}
            #     _t = 0
            #     for t, semantic_seg_t in self.semantic_segmentations.items():
            #         if t in frame_idxes:
            #             subseq_semantic_segmentation[_t] = semantic_seg_t
            #             _t += 1
            #     subseq_dict["semantic_segmentations"] = subseq_semantic_segmentation
        
        if self.ignore_masks is not None:
            subseq_dict["ignore_masks"] =  [self.ignore_masks[t] for t in frame_idxes]
        
        return self.__class__(subseq_dict, self.path_to_images, serialize=True)
