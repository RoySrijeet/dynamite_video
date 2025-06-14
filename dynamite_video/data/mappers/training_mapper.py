import torch
import numpy as np

from collections import defaultdict

from dynamite_video.data.utils.clicker import get_clicks_coords
from dynamite_video.data.utils.data_utils import (
    apply_color_augmentation, 
    apply_random_flip,
    apply_resize_scale,
)

class TrainingMapper:
    """
    A callable used by the dataloader, which takes a sample dictionary,
    and maps it into a format used by DynaMITe-Video.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # print("Training Mapper")
    
    
    def __call__(self, sample):
        """
        Prepare a sample (training clip) for training forward pass.

        Args:
            sample: a dictionary with the following keys:
                `vid_id`: name of the source video sequence
                `ref_frame`: first ferame of the clip
                `other_frames`: list of rest of the frames in the clip
                `video`: source video cast as GenericVideoSequence object

        """
       # source video
        video = sample['video']
        # get absolute frame indices in the video that will form a clip
        frame_indices = [sample["ref_frame"]] + sample["other_frames"]
        clip_id = sample["vid_id"]

        # extract clip from the video
        clip = video.extract_subsequence(frame_indices, new_id=clip_id)

        # load images [T,H,W,3]
        images = clip.load_images()
        
        # load binary instance masks [T,N,H,W]
        binary_masks, targets_per_frame, target_ids, ignore_masks = clip.prepare_masks()
        
        # color augmentations
        if self.cfg.INPUT.AUGMENTATION.COLOR_AUG:
            images = apply_color_augmentation(images)

        # data augmentations
        images, binary_masks, ignore_masks = apply_random_flip(images, 
                                                                binary_masks,
                                                                ignore_masks,
                                                                axis=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_AXIS,
                                                                prob=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_PROB,
                                                            )
        
        images, binary_masks, ignore_masks, padding_mask = apply_resize_scale(images,
                                                                binary_masks,
                                                                ignore_masks,
                                                                min_scale=self.cfg.INPUT.AUGMENTATION.MIN_SCALE,
                                                                max_scale=self.cfg.INPUT.AUGMENTATION.MAX_SCALE,
                                                                target_dims=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE,
                                                            )
        
        images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
        if self.cfg.INPUT.RGB:
            # BGR -> RGB
            images = np.flip(images, 1).copy()
        
        # semantic maps
        T, _, H, W = binary_masks.shape
        # for each target, keep a record of all the frames it appears in
        frame_target_occupancy = defaultdict(list)
        semantic_masks = np.zeros((T,H,W), dtype=np.uint8)
        for fr_idx in range(T):
            for inst_id, inst_mask in enumerate(binary_masks[fr_idx]):
                semantic_masks[fr_idx][inst_mask==1] = inst_id+1
                if np.any(inst_mask):
                    # target is present in the frame
                    frame_target_occupancy[inst_id+1].append(fr_idx)
        
        # NOTE: 0-labelled region in semantic masks at this point 
        # corresponds to the ignore mask area and the padding area
        
        # background mask contains ignore mask and all small objects and gaps that were omitted
        bg_masks = np.logical_not(np.logical_or(padding_mask, semantic_masks)).astype('uint8')
        # set all of background as ignore mask
        ignore_masks = bg_masks
        
        # sample clicks
        num_clicks_per_target, fg_coords_list, bg_coords_list, max_timestamp_list = get_clicks_coords(
                                                                                    target_ids=target_ids,
                                                                                    target_masks=binary_masks, 
                                                                                    bg_masks=None,
                                                                                    frame_target_occupancy=frame_target_occupancy,
                                                                                    serial_to_orig_id=clip.serial_to_orig_id,
                                                                                    max_instances_per_category=clip.meta_info["max_instances_per_category"],
                                                                                    max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                                                    optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                                                    bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                                                    gamma=self.cfg.CLICKER.TRAINING.GAMMA,
                                                                                    start_t=1,
                                                                                )
        if not all(np.sum(num_clicks_per_target, axis=0)):
            raise "One or more targets did not receive a click!"

        meta_info = {
            "orig_dims": images.shape[2:],
            "seq_name": video.id,
            "orig_to_serial_id": clip.orig_to_serial_id, 
            "serial_to_orig_id": clip.serial_to_orig_id, 
            "ignore_class": clip.ignore_class,
            "target_categories": clip.object_categories,
            "max_instances_per_category": clip.meta_info['max_instances_per_category'],
        }
        
        return {
            "images": torch.as_tensor(images, dtype=torch.uint8),
            "binary_masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
            "semantic_masks": torch.as_tensor(semantic_masks, dtype=torch.uint8),
            "padding_mask": torch.as_tensor(padding_mask, dtype=torch.uint8),
            "bg_masks": torch.as_tensor(bg_masks, dtype=torch.uint8),
            "ignore_masks": torch.as_tensor(ignore_masks, dtype=torch.bool),
            "objects_per_frame": targets_per_frame,
            "num_clicks_per_object": num_clicks_per_target,
            "fg_coords_list": fg_coords_list,
            "bg_coords_list": bg_coords_list,
            "max_timestamp_list": max_timestamp_list,
            "meta": meta_info,
        }