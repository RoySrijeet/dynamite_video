import numpy as np

from dynamite_video.data.utils.data_utils import (
    apply_color_augmentation, 
    apply_resizer,
    apply_random_horizontal_flip,
    apply_random_crop
)

from dynamite_video.data.utils.clicker import get_clicks_coords

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
                `ref_inst_ids`: instances present in the clip
                `video`: source video cast as GenericVideoSequence object

        """
        # source video
        video = sample['video']
        # get absolute frame indices in the video that will form a clip
        frame_indices = [sample["ref_frame"]] + sample["other_frames"]
        clip_inst_ids = sample["ref_inst_ids"]
        clip_id = sample["vid_id"]

        # extract clip from the video
        clip = video.extract_subsequence(frame_indices, clip_inst_ids, clip_id)

        # load RGB frames as a list of np.ndarrays, each [H, W, 3]
        images = clip.load_images()
        
        # load binary and semantic masks as lists
        (
            masks,
            num_instances_per_frame,
            instance_ids
        ) = clip.prepare_masks()

        # color augmentations
        if self.cfg.INPUT.AUGMENTATION.COLOR_AUG:
            images = apply_color_augmentation(images)

        images = np.stack(images)                                       # [T, H, W, 3]
        masks = np.stack([np.stack(masks_t) for masks_t in masks])      # [T, N, H, W]

        meta_info = {
            "orig_dims": images[0].shape[:2],
            "seq_name": video.id,
            "frame_indices": frame_indices,
            "orig_to_serial_id": clip.orig_to_serial_id, 
            "serial_to_orig_id": clip.serial_to_orig_id,
        }

        # data augmentations
        # apply resizing
        images, masks = apply_resizer(images, 
                                    masks, 
                                    mode="min_dim",
                                    min_dim=self.cfg.INPUT.AUGMENTATION.MIN_DIM_TRAIN,
                                    max_dim=self.cfg.INPUT.AUGMENTATION.MAX_DIM_TRAIN,
                                )
        # apply horizontal flip
        if self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP:
            images, masks = apply_random_horizontal_flip(images, 
                                                        masks,
                                                        flip_axis=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_AXIS,
                                                        prob=self.cfg.INPUT.AUGMENTATION.RANDOM_FLIP_PROB,
                                            )
        # apply random crops
        if self.cfg.INPUT.AUGMENTATION.RANDOM_CROP:
            crop_size=self.cfg.INPUT.AUGMENTATION.IMAGE_SIZE
            (
                images, 
                masks, 
                semantic_masks, # [T, H, W]
                padding_mask, 
                frame_instance_occupancy
            ) = apply_random_crop(images, 
                                masks, 
                                instance_ids,
                                crop_size=crop_size,
                                MIN_MASK_AREA=self.cfg.TRAINING.MIN_MASK_AREA,
                            )
        
        bg_masks = np.logical_not(np.logical_or(padding_mask, semantic_masks)).astype('uint8')  # [T, H, W]

        images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
        if self.cfg.INPUT.RGB:
            # BGR -> RGB
            images = np.flip(images, 1).copy()

        while True:
            num_clicks_per_object, fg_coords_list, bg_coords_list, max_timestamp_list = get_clicks_coords(
                                                                                        instance_ids=instance_ids,
                                                                                        instance_masks=masks, 
                                                                                        bg_masks=bg_masks,
                                                                                        frame_instance_occupancy=frame_instance_occupancy,
                                                                                        max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                                                        first_click_center=self.cfg.CLICKER.TRAINING.FIRST_CLICK_CENTER,
                                                                                        optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                                                        bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                                                        gamma=self.cfg.CLICKER.TRAINING.GAMMA,
                                                                                        start_t=1,
                                                                                    )
            if all(np.sum(num_clicks_per_object, axis=0)):
                break
            else:
                # np.save("/home/roy/REPOS/dynamite_video/notebooks/kitti_step/problem_mask.npy", masks)
                # np.save("/home/roy/REPOS/dynamite_video/notebooks/kitti_step/problem_sem_mask.npy", semantic_masks)
                raise "One or more instances did not receive a click!"


        return {
            "images": images,
            "instance_masks": masks,
            "semantic_masks": semantic_masks,
            "padding_mask": padding_mask,
            "bg_masks": bg_masks,
            "instance_ids": instance_ids,
            "num_instances_per_frame": num_instances_per_frame,
            "frame_instance_occupancy": frame_instance_occupancy,
            "ref_frame_index": 0,
            "num_clicks_per_object": num_clicks_per_object,
            "fg_coords_list": fg_coords_list,
            "bg_coords_list": bg_coords_list,
            "max_timestamp_list": max_timestamp_list,
            "meta": meta_info,
        }