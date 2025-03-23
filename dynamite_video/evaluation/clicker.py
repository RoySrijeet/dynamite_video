import torch
import random
import numpy as np

from dynamite_video.data.utils.clicker import get_center_coords

class SequenceManager:
    """
    Click manager
    """

    def __init__(self, metadata):
        """
        Initialize with information on the sequence data, including ground truth masks
        """

        self.sequence_id = metadata["id"]
        self.sequence_length = metadata["length"]
        self.orig_dims = metadata["orig_dims"]
        
        # arrays
        self.images = metadata["images"]                    # [T,3,H,W]
        self.instance_masks = metadata["instance_masks"]    # [T,N,H,W]
        self.semantic_maps = metadata["semantic_maps"]      # [T,H,W]
        self.bg_masks = metadata["bg_masks"]                # [T,H,W]
        self.padding_mask = metadata["padding_mask"]        # [H,W]
        
        # in which frame did each instance first appear
        self.instance_discovery = metadata["instance_discovery"]
        self.instances = sorted(self.instance_discovery.keys())
        # IDs of instances present in each frame
        self.instances_per_frame = metadata["instances_per_frame"]
        
        # mappings between original instance IDs and serial IDs
        self.orig_to_serial_ids = metadata["orig_to_serial_ids"]
        self.serial_to_orig_ids = metadata["serial_to_orig_ids"]

        self.clip_indices = metadata["indices"]
        self.clip_length = metadata["clip_length"]
        self.num_overlapping_frames = metadata["num_overlapping_frames"]
        
        # click radius - region around a existing click that is excluded when sampling a new click
        self.click_radius = 5
        # Strategy to avoid regions while sampling next clicks
        # 0: new click avoids all the previously sampled click locations
        # 1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks
        self.sampling_strategy = 1
        
        
        # initialize buffers
        # foreground clicks sampled on each object in each frame
        self.fg_coords_list = [[[] for _ in range(self.num_instances)] for _ in range(self.sequence_length)]
        # background clicks sampled on each frame
        self.bg_coords_list = [[] for _ in range(self.sequence_length)]
        # time stamp of the latest click on each frame
        self.max_timestamps = [0 for _ in range(self.sequence_length)]
        # num clicks on each object in each frame
        self.num_clicks_per_object = np.zeros((self.sequence_length, self.num_instances)).astype('int').tolist()

        # get initial clicks on object center
        self.get_gt_clicks()
        
        self.pred_masks = [[] for _ in range(self.sequence_length)]

    
    @property
    def num_instances(self):
        return len(self.instances)
    

    def get_gt_clicks(self):
        """
        For each instance present in the sequence, go to the frame where the 
        instance first appeared (specified by `self.instance_discovery`) and 
        sample a foreground click at the center of the instance

        Populates click buffers with sampled click information
        """
        t = 1
        for inst_id, fr_idx in self.instance_discovery.items():
            # Instance with ID `inst_id` first appeared in frame at index `fr_idx``
            
            # binary segmentation mask (ground truth)
            inst_mask = self.instance_masks[fr_idx][inst_id-1]

            # center coordinates of the foreground mask
            center_coords = get_center_coords(inst_mask)

            # record sampled click
            self.fg_coords_list[fr_idx][inst_id-1].append([center_coords[0], center_coords[1], inst_id-1, fr_idx, t])
            self.num_clicks_per_object[fr_idx][inst_id-1] += 1
            self.max_timestamps[fr_idx] = t
            t+=1


    
    def extract_clip(self, indices):
        """
        Given a list of frame indices, extract a clip consisting of these indices 
        from the whole sequence

        Args:
            indices: list(int)

        Returns:
            clip: dict. The format should be consistent with that of the batch input for 
                training pass, i.e., inputs argument in `DynamiteModel.forward()`
        """
        clip = {}
        clip["images"] = self.images[indices[0]:indices[-1]+1]
        clip["num_instances_per_frame"] = [len(self.instances_per_frame[fr_idx]) for fr_idx in indices]
        clip["num_clicks_per_object"] = self.num_clicks_per_object[indices[0]:indices[-1]+1]
        clip["fg_coords_list"] = self.fg_coords_list[indices[0]:indices[-1]+1]
        clip["bg_coords_list"] = self.bg_coords_list[indices[0]:indices[-1]+1]
        clip["max_timestamp_list"] = self.max_timestamps[indices[0]:indices[-1]+1]

        # in case no foreground clicks are found on a given instance, simulate 
        # some from the predicted masks of the overlapping frames
        if not all(np.sum(clip["num_clicks_per_object"], axis=0)):
            
            overlapping_frame_indices = indices[:self.num_overlapping_frames]
            overlapping_frame_preds = torch.stack(self.pred_masks[overlapping_frame_indices[0]:overlapping_frame_indices[-1]+1])
            t = max(clip["max_timestamp_list"]) + 1

            # instances that didn't get a click
            click_counts = np.sum(clip["num_clicks_per_object"], axis=0)
            for inst_id, cc in enumerate(click_counts):
                if cc>0:
                    continue
                inst_masks = overlapping_frame_preds[:,inst_id]
                choice_range = list(range(inst_masks.shape[0]))
                while True:
                    if len(choice_range) == 0:
                        break
                    choice = random.sample(choice_range, 1)[0]
                    choice_range.remove(choice)
                    if inst_masks[choice].any():
                        center_coords = get_center_coords(inst_masks[choice])
                        fr_idx = overlapping_frame_indices[choice]
                        self.fg_coords_list[fr_idx][inst_id].append([center_coords[0], center_coords[1], inst_id, fr_idx, t])
                        self.num_clicks_per_object[fr_idx][inst_id] += 1
                        self.max_timestamps[fr_idx] = t
                        t += 1
                        break
        return clip
    

    def save_pred_masks(self, pred_masks, indices):
        for idx, pred in zip(indices, pred_masks):
            self.pred_masks[idx] = pred
    
        



    