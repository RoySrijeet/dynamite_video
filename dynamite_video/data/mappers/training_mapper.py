import torch
import numpy as np

from collections import defaultdict

from dynamite_video.data.utils.clicker import get_clicks_coords
from dynamite_video.data.utils.data_utils import (
    apply_color_augmentation, 
    apply_random_flip,
    apply_resize_scale,
    apply_random_crop,
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

        # load RGB frames (cv2 loads in BGR format) as a list of np.ndarrays, each of shape [H, W, 3]
        images = clip.load_images()
        
        # load binary instance masks
        instance_masks, instances_per_frame, instance_ids = clip.prepare_masks()
        
        # color augmentations
        if self.cfg.INPUT.AUGMENTATION.COLOR_AUG:
            images = apply_color_augmentation(images)

        images = np.stack(images)                                                           # [T, H, W, 3]
        instance_masks = np.stack([np.stack(fr_masks) for fr_masks in instance_masks])      # [T, N, H, W]

        meta_info = {
            "orig_dims": images.shape[1:3],
            "seq_name": video.id,
            "frame_indices": frame_indices,
            "orig_to_serial_id": clip.orig_to_serial_id, 
            "serial_to_orig_id": clip.serial_to_orig_id,
        }
        
        # data augmentations
        if self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP:
            images, instance_masks = apply_random_flip(images, 
                                                        instance_masks,
                                                        axis=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_AXIS,
                                                        prob=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_PROB,
                                                    )
        
        images, instance_masks = apply_resize_scale(images,
                                                    instance_masks,
                                                    min_scale=self.cfg.INPUT.AUGMENTATION.MIN_SCALE,
                                                    max_scale=self.cfg.INPUT.AUGMENTATION.MAX_SCALE,
                                                    target_dims=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE,
                                                )
        
        images, instance_masks, padding_mask = apply_random_crop(images, 
                                                                instance_masks, 
                                                                crop_size=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE,
                                                            )
        
        images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
        if self.cfg.INPUT.RGB:
            # BGR -> RGB
            images = np.flip(images, 1).copy()
        
        # semantic maps
        T, _, H, W = instance_masks.shape
        # for each instance, keep a record of all the frames it appears in
        frame_instance_occupancy = defaultdict(list)
        semantic_masks = []
        for fr_idx in range(T):
            map = np.zeros((H,W))
            for inst_id, inst_mask in enumerate(instance_masks[fr_idx]):
                map[inst_mask==1] = inst_id+1
                if np.any(inst_mask):
                    # instance is present in the frame
                    frame_instance_occupancy[inst_id+1].append(fr_idx)
            semantic_masks.append(map)
        semantic_masks = np.stack(semantic_masks).astype('uint8')
        
        bg_masks = np.logical_not(np.logical_or(padding_mask, semantic_masks)).astype('uint8')  # [T, H, W]
        
        # sample clicks
        num_clicks_per_object, fg_coords_list, bg_coords_list, max_timestamp_list = get_clicks_coords(
                                                                                    instance_ids=instance_ids,
                                                                                    instance_masks=instance_masks, 
                                                                                    bg_masks=bg_masks,
                                                                                    frame_instance_occupancy=frame_instance_occupancy,
                                                                                    max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                                                    first_click_center=self.cfg.CLICKER.TRAINING.FIRST_CLICK_CENTER,
                                                                                    optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                                                    bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                                                    gamma=self.cfg.CLICKER.TRAINING.GAMMA,
                                                                                    start_t=1,
                                                                                )
        if not all(np.sum(num_clicks_per_object, axis=0)):
            raise "One or more instances did not receive a click!"

        return {
            "images": torch.as_tensor(images, dtype=torch.uint8),
            "instance_masks": torch.as_tensor(instance_masks, dtype=torch.uint8),
            "semantic_masks": torch.as_tensor(semantic_masks, dtype=torch.uint8),
            "padding_mask": torch.as_tensor(padding_mask, dtype=torch.uint8),
            "bg_masks": torch.as_tensor(bg_masks, dtype=torch.uint8),
            "instance_ids": instance_ids,
            "instances_per_frame": instances_per_frame,
            "frame_instance_occupancy": dict(frame_instance_occupancy),
            "num_clicks_per_object": num_clicks_per_object,
            "fg_coords_list": fg_coords_list,
            "bg_coords_list": bg_coords_list,
            "max_timestamp_list": max_timestamp_list,
            "meta": meta_info,
        }
    

    # vis
    # def __callvis__(self, sample):
    #     """
    #     Prepare a sample (training clip) for training forward pass.

    #     Args:
    #         sample: a dictionary with the following keys:
    #             `vid_id`: name of the source video sequence
    #             `ref_frame`: first ferame of the clip
    #             `other_frames`: list of rest of the frames in the clip
    #             `ref_inst_ids`: instances present in the clip
    #             `video`: source video cast as GenericVideoSequence object

    #     """
    #     # source video
    #     video = sample['video']
    #     # get absolute frame indices in the video that will form a clip
    #     frame_indices = [sample["ref_frame"]] + sample["other_frames"]
    #     clip_inst_ids = sample["ref_inst_ids"]  # instances present in ref_frame
    #     clip_id = sample["vid_id"]

    #     # extract clip from the video
    #     clip = video.extract_subsequence(frame_indices, clip_inst_ids, clip_id)

    #     # load RGB frames (cv2 loads in BGR format) as a list of np.ndarrays, each of shape [H, W, 3]
    #     images = clip.load_images()
        
    #     # load binary instance masks
    #     instance_masks, instances_per_frame, instance_ids = clip.prepare_masks()
        
    #     # color augmentations
    #     if self.cfg.INPUT.AUGMENTATION.COLOR_AUG:
    #         images = apply_color_augmentation(images)

    #     images = np.stack(images)                                                           # [T, H, W, 3]
    #     instance_masks = np.stack([np.stack(fr_masks) for fr_masks in instance_masks])      # [T, N, H, W]

    #     meta_info = {
    #         "orig_dims": images.shape[1:3],
    #         "seq_name": video.id,
    #         "frame_indices": frame_indices,
    #         "orig_to_serial_id": clip.orig_to_serial_id, 
    #         "serial_to_orig_id": clip.serial_to_orig_id,
    #     }
        
    #     # data augmentations
    #     if self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP:
    #         images, instance_masks = apply_random_flip(images, 
    #                                                     instance_masks,
    #                                                     axis=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_AXIS,
    #                                                     prob=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_PROB,
    #                                                 )
        
    #     images, instance_masks = apply_resize_scale(images,
    #                                                 instance_masks,
    #                                                 min_scale=self.cfg.INPUT.AUGMENTATION.MIN_SCALE,
    #                                                 max_scale=self.cfg.INPUT.AUGMENTATION.MAX_SCALE,
    #                                                 target_dims=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE,
    #                                             )
        
    #     images, instance_masks, padding_mask = apply_random_crop(images, 
    #                                                             instance_masks, 
    #                                                             crop_size=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE,
    #                                                         )
        
    #     images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
    #     if self.cfg.INPUT.RGB:
    #         # BGR -> RGB
    #         images = np.flip(images, 1).copy()
        
    #     # semantic maps
    #     T, _, H, W = instance_masks.shape
    #     # for each instance, keep a record of all the frames it appears in
    #     frame_instance_occupancy = defaultdict(list)
    #     semantic_masks = []
    #     for fr_idx in range(T):
    #         map = np.zeros((H,W))
    #         for inst_id, inst_mask in enumerate(instance_masks[fr_idx]):
    #             map[inst_mask==1] = inst_id+1
    #             if np.any(inst_mask):
    #                 # instance is present in the frame
    #                 frame_instance_occupancy[inst_id+1].append(fr_idx)
    #         semantic_masks.append(map)
    #     semantic_masks = np.stack(semantic_masks).astype('uint8')
        
    #     bg_masks = np.logical_not(np.logical_or(padding_mask, semantic_masks)).astype('uint8')  # [T, H, W]
        
    #     # sample clicks
    #     num_clicks_per_object, fg_coords_list, bg_coords_list, max_timestamp_list = get_clicks_coords(
    #                                                                                 instance_ids=instance_ids,
    #                                                                                 instance_masks=instance_masks, 
    #                                                                                 bg_masks=bg_masks,
    #                                                                                 frame_instance_occupancy=frame_instance_occupancy,
    #                                                                                 max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
    #                                                                                 first_click_center=self.cfg.CLICKER.TRAINING.FIRST_CLICK_CENTER,
    #                                                                                 optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
    #                                                                                 bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
    #                                                                                 gamma=self.cfg.CLICKER.TRAINING.GAMMA,
    #                                                                                 start_t=1,
    #                                                                             )
    #     if not all(np.sum(num_clicks_per_object, axis=0)):
    #         raise "One or more instances did not receive a click!"

    #     return {
    #         "images": torch.as_tensor(images, dtype=torch.uint8),
    #         "instance_masks": torch.as_tensor(instance_masks, dtype=torch.uint8),
    #         "semantic_masks": torch.as_tensor(semantic_masks, dtype=torch.uint8),
    #         "padding_mask": torch.as_tensor(padding_mask, dtype=torch.uint8),
    #         "bg_masks": torch.as_tensor(bg_masks, dtype=torch.uint8),
    #         "instance_ids": instance_ids,
    #         "instances_per_frame": instances_per_frame,
    #         "frame_instance_occupancy": dict(frame_instance_occupancy),
    #         "ref_frame_index": 0,
    #         "num_clicks_per_object": num_clicks_per_object,
    #         "fg_coords_list": fg_coords_list,
    #         "bg_coords_list": bg_coords_list,
    #         "max_timestamp_list": max_timestamp_list,
    #         "meta": meta_info,
    #     }