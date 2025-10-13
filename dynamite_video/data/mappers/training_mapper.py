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
        # source video, a `GenericVideoSequence`
        video = sample['video']

        # get absolute frame indices in the video that will form a clip
        frame_indices = [sample["ref_frame"]] + sample["other_frames"]

        # extract clip from the video as a `GenericVideoSequence`
        clip = video.extract_subsequence(frame_indices, new_id=sample["vid_id"])

        # load images [T,H,W,3]
        images = clip.load_images()
        
        # load binary instance masks [T,N,H,W]; NOTE: ignore masks are not part of the binary masks
        binary_masks, ignore_masks = clip.prepare_masks()
        
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
        
        # foreground masks
        T, _, H, W = binary_masks.shape
        panoptic_masks = np.zeros((T,H,W), dtype=np.uint8)      # T,H,W
        for fr_idx in range(T):
            for inst_id, inst_mask in enumerate(binary_masks[fr_idx]):
                panoptic_masks[fr_idx][inst_mask==1] = inst_id+1
        
        # NOTE: at this point, 0-labelled regions in foreground masks correspond
        # to the ignore mask area, the padding area, and any potential background
        
        # consider all parts of the image without fg label as bg
        bg_masks = (panoptic_masks==0) & ~padding_mask          # T,H,W

        if ignore_masks is None:
            ignore_masks = np.zeros_like(panoptic_masks)
        
        # sample clicks
        (num_clicks_per_target, 
         fg_coords_list, 
         bg_coords_list, 
         max_timestamp_list) = get_clicks_coords(target_masks=binary_masks, 
                                                max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                bg_masks=bg_masks,
                                                bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                gamma=self.cfg.CLICKER.TRAINING.GAMMA,
                                                start_t=1,
                                            )
        if not all(np.sum(num_clicks_per_target, axis=0)):
            raise "One or more targets in this clip did not receive a click!"

        meta_info = {
            "orig_dims": video.image_dims,
            "seq_name": video.id,
            "frame_indices": frame_indices,
            "orig_to_serial_id": clip.orig_to_serial_id, 
            "serial_to_orig_id": clip.serial_to_orig_id, 
            "ignore_class": clip.ignore_class,
            "target_categories": clip.object_categories,
            "max_instances_per_category": clip.meta_info.get('max_instances_per_category', None),
        }
        
        return {
            # T,3,H,W image tensors, not normalized, padded region has value 128
            "images": torch.as_tensor(images, dtype=torch.uint8),
            
            # T,N,H,W binary masks of target objects. This includes the semantic maps of the `stuff` classes 
            # and the instance maps of `thing` class instances. Which means, the binary masks do not include 
            # region of `thing` class that is not covered by corresponding instances, and the VOID class.
            "binary_masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
            
            # H,W boolean padding mask where padded region is labeled True
            "padding_mask": torch.as_tensor(padding_mask, dtype=torch.bool),
            
            # T,H,W boolean mask for `VOID` class
            "ignore_masks": torch.as_tensor(ignore_masks, dtype=torch.bool),
            
            # T,H,W bg mask includes any region not covered by the target object binary masks
            # NOTE: this includes ignore mask regions as well
            "bg_masks": torch.as_tensor(bg_masks, dtype=torch.uint8),
            
            # T,H,W semantic map with serialized target IDs. Background gets labeled 0
            "panoptic_masks": torch.as_tensor(panoptic_masks, dtype=torch.uint8),

            # T,N array recording num clicks on each target in each frame
            "num_clicks_per_object": num_clicks_per_target,
            
            # list of fg clicks sampled on the clip. Each click follows the format: [y,x,i,f,t]
            "fg_coords_list": fg_coords_list,
            
            # list of bg clicks sampled on the clip. Each click follows the format: [y,x,-1,f,t]
            "bg_coords_list": bg_coords_list,

            # list of length T recording timestamp of the latest click on each frame
            "max_timestamp_list": max_timestamp_list,

            # info about the clip and its source video
            "meta": meta_info,
        }