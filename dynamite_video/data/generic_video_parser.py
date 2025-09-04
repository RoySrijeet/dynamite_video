# Adapted from https://github.com/Ali2500/TarViS

import cv2
import json
import numpy as np
import os

from collections import defaultdict, OrderedDict
from typing import Any, List, Dict, Union

from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.data.utils.data_utils import serialize_object_ids, decode_mask


def parse_generic_video_dataset(
        path_to_images: str, 
        annotations_content: Union[str, Dict[str, Any]],
        serialize: bool=False
):
    """
    Cast each sequence in the dataset to `GenericVideoSequence` class template

    Args:
        path_to_images: path to dataset images
        annotations_content: dataset annotation, may be a path to a JSON file
            or a dict with mask information
        serialize: boolean, whether to serialize object IDs or not
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
        # sanity check: object IDs in "segmentations" must match those in "categories"
        for seq in dataset["sequences"]:
            if "categories" in seq:
                
                seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
                assert seg_iids == set(seq["categories"].keys()), "Object ID mismatch in seq {}: {} vs. {}".format(
                    seq["id"], seg_iids, set(seq["categories"].keys())
                )

    # wrap each video sequence as a `GenericVideoSequence` object
    seqs = [GenericVideoSequence(seq, path_to_images, meta_info, serialize) for seq in dataset["sequences"]]

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
            meta_info,
            serialize=False
    ):
        """
        Args:
            seq_dict: dictionary containing info (resolution, annotation) of a 
                single video sequence from a dataset
            path_to_images: path to dataset images
            serialize: serialize non-sequential object ids
        """
        assert len(seq_dict["image_paths"]) == len(seq_dict["segmentations"])

        self.path_to_images = path_to_images
        self.image_paths = seq_dict["image_paths"]
        self.image_dims = (seq_dict["height"], seq_dict["width"])
        self.id = seq_dict["id"]
        self.meta_info = meta_info

        if not serialize:
            self.segmentations = seq_dict["segmentations"]
            self.semantic_segmentations = seq_dict.get("semantic_segmentations", None)
            self.object_categories = seq_dict["categories"]
            obj_ids = sorted(seq_dict["categories"].keys())
            self.orig_to_serial_id = OrderedDict(zip(obj_ids, obj_ids))
            self.serial_to_orig_id = OrderedDict(zip(obj_ids, obj_ids))

        # serialize non-sequential object IDs
        else:
            # obtain mappings between original and sequential IDs
            self.orig_to_serial_id, self.serial_to_orig_id = serialize_object_ids(sorted(seq_dict["categories"].keys()))
            if all(key == value for key, value in self.orig_to_serial_id.items()):
                # already serial
                self.segmentations = seq_dict["segmentations"]
                self.semantic_segmentations = seq_dict.get("semantic_segmentations", None)
                self.object_categories = seq_dict["categories"]
            else:
                # serialize seg keys and labels
                segmentations = seq_dict["segmentations"]
                semantic_segmentations = seq_dict.get("semantic_segmentations", None)
                self.segmentations, self.semantic_segmentations = self.serialize_masks(segmentations, semantic_segmentations, self.orig_to_serial_id)
                # update IDs in category map
                self.object_categories = {self.orig_to_serial_id.get(obj_id): value for obj_id, value in seq_dict["categories"].items()}
        
        self.ignore_masks = seq_dict.get("ignore_masks", None)
        self.ignore_class = self.meta_info.get("ignore_class", None)
        self.object_areas = None
        self.fpack_reader = None
    
    
    def serialize_masks(self, segmentations, semantic_segmentations, orig_to_serial_id):
        """
        Rename annotation keys from original object IDs to sequential IDs

        Args:
            segmentations: dictionary of object masks where original object IDs
                were used as keys
            semantic_segmentation: dictionary of semantic masks where original
                object IDs are used as pixel labels
            orig_to_serial_ids: mapping from original IDs to sequential IDs
        """
        
        # update object masks
        updated_segmentations = []
        for fr_idx, segs_t in enumerate(segmentations):
            # update keys
            new_seg = {orig_to_serial_id.get(obj_id): value for obj_id, value in segs_t.items()}
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
    def object_ids(self):
        """IDs of objects present in the video"""
        return sorted(self.object_categories.keys())
    
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
    

    def prepare_masks(self):
        """
        Decode RLE masks to numpy arrays of binary segmentation masks

        Returns:
            binary_masks: binary segmentation masks of the target objects in the frames of the video
                as a T,N,H,W `np.ndarray`
            ignore_masks: ignore masks, if present, as a T,H,W `np.ndarray`
        """
        zero_mask = np.zeros(self.image_dims).astype('uint8')
        
        # binary segmentation masks
        binary_masks = []
        for fr_masks in self.segmentations:
            binary_masks_fr = []
            
            # add one channel for each target object present in the clip
            for obj_id in self.object_ids:
                if obj_id in fr_masks:
                    # decode RLE
                    rle = fr_masks[obj_id]
                    img_dims = None if isinstance(rle, dict) else self.image_dims
                    _m = decode_mask(rle, img_dims)
                    binary_masks_fr.append(_m)
                else:
                    # record an empty mask if target object is not present
                    binary_masks_fr.append(zero_mask)

            binary_masks.append(np.stack(binary_masks_fr))  # N,H,W
        binary_masks = np.stack(binary_masks)               # T,N,H,W
        
        # ignore masks
        ignore_masks = None
        if self.ignore_masks is not None:
            ignore_masks = [decode_mask(ig_msk, img_dims) for ig_msk in self.ignore_masks]
            ignore_masks = np.stack(ignore_masks)
        
        return binary_masks, ignore_masks

    
    def prepare_eval_masks(self, orig_to_serial_id, fill_value=0):
        """
        Prepare ground truth panoptic masks for evaluation. NOTE: `thing` classes are
        already excluded

        The binary RLE masks are converted to the panoptic mask

        Args:
            fill_value: any region not covered by the binary RLEs is assigned the `fill_value`
        """
        # store where each target appears for the first time
        object_appearance = defaultdict(list)
        # store which targets have already been discovered
        object_discovery = set()
        
        # read the panoptic masks with serial IDs of the targets
        panoptic_masks = np.full((len(self), self.height, self.width), fill_value=fill_value, dtype=np.uint8)
        for fr_idx, fr_rles in enumerate(self.segmentations):
            for obj_id in fr_rles:
                # decode RLE
                img_dims = None if isinstance(fr_rles[obj_id], dict) else self.image_dims
                _m = decode_mask(fr_rles[obj_id], img_dims)
                
                serial_id = orig_to_serial_id[obj_id]
                panoptic_masks[fr_idx][np.where(_m==1)] = serial_id
                
                # store which frame an object first appears
                if serial_id not in object_discovery:
                    object_appearance[fr_idx].append(serial_id)

                # mark the object discovered
                object_discovery.add(serial_id)
        
        ignore_masks = np.zeros_like(panoptic_masks, dtype=np.bool_)
        for fr_idx, ign_rles in enumerate(self.ignore_masks):
            # decode RLE
            img_dims = None if isinstance(ign_rles, dict) else self.image_dims
            _m = decode_mask(ign_rles, img_dims)
            ignore_masks[fr_idx][np.where(_m==1)] = True

        return panoptic_masks, ignore_masks, object_appearance

    
    def extract_subsequence(self, frame_idxes: List[int], object_ids_to_keep: List[int]=None, new_id: str=""):
        """
        Extract the specified frames from the video and return it as a clip

        The clip is also cast as a `GenericVideoSequence` object

        Args:
            frame_idxes: frames to be extracted from the main video
            object_ids_to_keep: objects to keep
            new_id: new id for the subsequence

        Returns:
            A GenericVideoSequence object with the specified frames and objects
        """

        assert all([t in range(len(self)) for t in frame_idxes])

        subseq_dict = {
            "id": new_id if new_id else self.id,
            "height": self.image_dims[0],
            "width": self.image_dims[1],
            "image_paths": [self.image_paths[t] for t in frame_idxes],
        }
        
        if object_ids_to_keep is None:

            subseq_dict["segmentations"] = []
            subseq_objects = []
            for fr_idx in frame_idxes:
                subseq_dict["segmentations"].append(self.segmentations[fr_idx])
                subseq_objects.extend(self.segmentations[fr_idx].keys())

            subseq_objects = sorted(set(subseq_objects))
            subseq_dict["categories"] = {iid: self.object_categories[iid] for iid in subseq_objects}
        
        else:
            subseq_dict["segmentations"] = [
                {
                    iid: segmentations_t[iid]
                    for iid in segmentations_t if iid in object_ids_to_keep
                }
                for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
            ]

            subseq_dict["categories"] = {iid: self.object_categories[iid] for iid in object_ids_to_keep}
        
        if self.ignore_masks is not None:
            subseq_dict["ignore_masks"] =  [self.ignore_masks[t] for t in frame_idxes]
        
        return self.__class__(subseq_dict, self.path_to_images, self.meta_info, serialize=True)
