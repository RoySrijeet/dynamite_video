import cv2
import numpy as np
import os
import torch

from collections import defaultdict
from PIL import Image
from typing import List, Mapping, Optional, Tuple

from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.evaluation.eval_utils import *

class SequenceManager:
    """
    Sequence manager
    """

    def __init__(
            self, 
            sequence: GenericVideoSequence, 
            dataset_meta: Mapping, 
            tfms: Mapping, 
            output_dir: str, 
            save_vis: bool
    ) -> None:
        """
        Initialize with information on the sequence data, including ground truth masks

        Args:
            sequence: `GenericVideoSequence` instance of current video
            dataset_meta: dict with dataset level info on following properties:
                * clip_length: int, length of each sub-sequence
                * num_overlapping_frames: int, overlap between consecutive sub-sequences
            tfms: transformation info from config, cfg.INPUT
            output_dir: str, path to the directory where visualizations are stored
            save_vis: bool, whether to save visualizations or not
        """
        # sequence level information
        
        # `GenericVideoSequence` instance of current video
        self.sequence = sequence
        # num of frames in the sequence
        self.T = len(self.sequence)
        # num of targets in the sequence (does not include `VOID` and `thing` classes)
        self.N = len(self.sequence.object_ids)
        # original spatial resolution
        self.orig_H, self.orig_W = self.sequence.image_dims

        # target level information
        
        self.orig_target_ids = self.sequence.object_ids
        # serialize target IDs, 1-indexed
        self.orig_to_serial_ids, self.serial_to_orig_ids = serialize_target_ids(sorted(self.orig_target_ids))
        # serial target ids
        self.target_ids = sorted(self.serial_to_orig_ids.keys())
        # bg label
        self.bg_id = 0
        # maintain a record to keep track of targets that were already discovered
        self.target_discovery = set()
        
        # load tensors
        
        # load images, T,orig_H,orig_W,3
        self.images = self.sequence.load_images()
        # load ground truth panoptic masks, T,orig_H,orig_W with serialized IDs
        self.gt_masks, self.ignore_masks, self.target_appearance = self.sequence.prepare_eval_masks(self.orig_to_serial_ids, self.bg_id)
        # apply transformation
        # `images` now has shape T,3,H,W; `gt_masks`, `ignore_masks` now has shape T,H,W
        self.H, self.W = self.compute_tfm_sizes(tfms)
        # bg mask - T,H,W; boolean
        self.bg_masks = (self.gt_masks==self.bg_id)

        # click level information
        
        # click radius - region around a existing click that is excluded when sampling a new click
        self.click_radius = 5
        # Strategy to avoid regions while sampling next clicks
        # 0: new click avoids all the previously sampled click locations
        # 1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks
        self.sampling_strategy = 1
        # maintain a map of regions to avoid during click sampling, initially all true
        self.not_clicked_map = np.ones_like(self.gt_masks).astype(np.bool_)
        # num clicks on each target in each frame
        self.num_clicks_per_target = np.zeros((self.T, self.N), dtype=np.uint16)
        # num clicks per frame
        self.num_clicks_per_frame = np.zeros((self.T), dtype=np.uint16)
        # time stamp of the latest click on each frame
        self.max_timestamps = np.zeros((self.T), dtype=np.uint16)
        # first click at timestamp 1
        self.t = 1
        # foreground clicks sampled on each target in each frame from the ground truth mask
        self.gt_fg_coords_list = [[] for _ in range(self.T)]
        # background clicks sampled on each frame from the ground truth mask
        self.gt_bg_coords_list = [[] for _ in range(self.T)]
        # foreground clicks sampled on each target from the predicted mask of some overlapping frame
        self.overlap_coords_list = [[] for _ in range(self.T)]
        
        # prediction information
        
        # to store predicted masks
        self.pred_masks = np.zeros((self.T, self.H, self.W), dtype=np.uint8)
        # to store prediction logits
        self.pred_logits = [[] for _ in range(self.T)]

        # rounding info
        self.round_num = 0

        # length of clips to be extracted from the sequence
        self.clip_length = dataset_meta["clip_length"]
        # overlap between successive clips
        self.num_overlapping_frames = dataset_meta["num_overlapping_frames"]
        
        # visualization paths
        self.output_dir = output_dir
        self.save_vis = save_vis
        if self.save_vis:
            # path to save predicted masks
            self.path_to_visualization = os.path.join(output_dir, "masks", self.sequence.id)
            os.makedirs(self.path_to_visualization)

        # I/O
        self.curr_clip_input = {}
        self.prev_clip_output = {}
        self.curr_overlapping_frames = None
        self.refine_frame = None

        # category labels
        self.category_labels = {}
        for orig_id, serial_id in self.orig_to_serial_ids.items():
            tgt_cls = orig_id // dataset_meta["max_instances_per_category"]
            label = dataset_meta["category_labels"][tgt_cls]
            inst_id = orig_id % dataset_meta["max_instances_per_category"]
            if inst_id != 0:
                label = label + "_" + str(inst_id)
            self.category_labels[serial_id] = label
        if self.save_vis:
            save_color_palette(self.category_labels, self.path_to_visualization)
        

    def compute_tfm_sizes(self, tfms: Mapping) -> Tuple[int, int]:
        """
        Compute transformed resolution

        Args:
            tfms: mapping containing transformation info from config, cfg.INPUT
        
        Returns:
            new_height, new_width: tuple, transformed height and width
        """
        new_height, new_width = self.orig_H, self.orig_W
        self.resize = False
        if tfms.AUGMENTATION.RESIZE_TEST:
            new_height, new_width = compute_resized_dims(self.orig_H, self.orig_W,
                                                        min_dim=tfms.AUGMENTATION.MIN_DIM_TEST,
                                                        max_dim=tfms.AUGMENTATION.MAX_DIM_TEST,
                                                    )
            self.images = resize_images(self.images, new_height, new_width)
            self.gt_masks = resize_masks(self.gt_masks, new_height, new_width, binary=False)
            self.ignore_masks = resize_masks(self.ignore_masks, new_height, new_width, binary=False)
            self.resize=True

        self.images = np.transpose(self.images, (0, 3, 1, 2))   # T,3,H,W
        if tfms.RGB:
            self.images = np.flip(self.images, 1).copy()        # BGR -> RGB
        return new_height, new_width
        

    def generate_clip_indices(self, start: int) -> List[List[int]]:
        """
        Given a start index, generate list of indices of clips from the sequence.
        If the start index is in the middle of the sequence, it generates clips in
        both forward and backward directions.

        Args:
            start: int, index of the first frame of the first clip
        
        Returns:
            indices: List[List[int]], list of frame indices per clip
        """
        start_copy = start
        indices = []
        step = self.clip_length - self.num_overlapping_frames
        if start < self.T - 1:
            while start + self.clip_length <= self.T:
                indices.append(list(range(start, start + self.clip_length)))
                start += step
            if len(indices) > 0 and indices[-1][-1] != self.T - 1:
                indices.append(list(range(indices[-1][-1] - self.num_overlapping_frames+1, self.T)))
            elif len(indices) == 0:
                indices.append(list(range(start, self.T)))
            indices[-1].append('_')

        # generate clips that go back in time
        if start_copy>0:
            bwd = []
            start_copy = min(start_copy, self.T-1)
            while start_copy - self.clip_length >= 0:
                bwd.append(list(range(start_copy, start_copy-self.clip_length, -1)))
                start_copy -= step
            if len(bwd) > 0 and bwd[-1][-1] != 0:
                bwd.append(list(range(bwd[-1][-1] + self.num_overlapping_frames-1, -1, -1)))
            elif len(bwd) == 0:
                bwd.append(list(range(start_copy, -1, -1)))
            indices.extend(bwd)
            indices[-1].append('_')
        self.overlap_coords_list = [[] for _ in range(self.T)]
        return indices

    
    def extract_clip(self, _indices: List[int], clip_idx: int) -> Mapping:
        """
        Prepare an input clip in the format the model forward pass expects

        If there is no overlap between clips, each clip (and targets in it) is handled independently. 
        For each target, clicks are sampled from its g.t. mask in the frame where it first appeared.
        
        If there is an overlap, clicks are sampled for each target from its predicted mask in the 
        overlapping frames. For any new target appearing in the intermediate clips, a click is sampled 
        from the ground truth mask of that target in the frame where it first appeared.

        Args:
            _indices: List[int], frame indices of the clip
        
        Returns:
            inputs: a mapping in the format expected by the model
        """

        indices, last_clip = self.get_serial_indices(_indices)
        
        # find targets in the clip
        clip_target_ids, frames_to_sample = self.find_targets_in_clip(indices)
        
        # serialize target IDs
        clip_orig_to_serial_id, clip_serial_to_orig_id = serialize_target_ids(clip_target_ids)

        # click info must be re-adjusted to be consistent with clip level frame indices and target ids
        clip_fg_coords_list, clip_bg_coords_list = [], []
        clip_num_clicks_per_target = np.zeros((len(indices), len(clip_target_ids)), dtype=np.uint16)
        clip_max_timestamps = [0 for _ in indices]
        t = self.t + 1
        
        for fr_idx, fr_targets in frames_to_sample.items():
            local_fr_idx = indices.index(fr_idx)
            
            # already sampled g.t. clicks - update the indices to clip-level values
            for fg_click in self.gt_fg_coords_list[fr_idx]:
                clk_y, clk_x, clk_i, clk_f, clk_t = fg_click
                local_tgt_id = clip_orig_to_serial_id[clk_i]
                clip_fg_coords_list.append([clk_y, clk_x, local_tgt_id, local_fr_idx, clk_t])
                clip_num_clicks_per_target[local_fr_idx][local_tgt_id-1] += 1
                clip_max_timestamps[local_fr_idx] += 1
            for bg_click in self.gt_bg_coords_list[fr_idx]:
                clk_y, clk_x, clk_i, clk_f, clk_t = bg_click
                clip_bg_coords_list.append([clk_y, clk_x, clk_i, local_fr_idx, clk_t])
                clip_max_timestamps[local_fr_idx] += 1
            
            # sample a click on the g.t. mask of the new target
            for tgt_id in fr_targets["new"]:
                # get target center coordinates
                center_coords = get_center_coords((self.gt_masks[fr_idx] == tgt_id).astype(np.uint8) * self.not_clicked_map[fr_idx])
                # center_coords = get_random_coords((self.gt_masks[fr_idx] == tgt_id).astype(np.uint8) * self.not_clicked_map[fr_idx], 1)[0]
                # record the click as [y,x,i,f,t]
                local_tgt_id = clip_orig_to_serial_id[tgt_id]
                clip_fg_coords_list.append([center_coords[0], center_coords[1], local_tgt_id, local_fr_idx, self.t])
                clip_num_clicks_per_target[local_fr_idx][local_tgt_id-1] += 1
                clip_max_timestamps[local_fr_idx] += 1
                self.record_gt_click(fr_idx, tgt_id, center_coords)
            
            # targets in overlapping frames
            if fr_targets["overlap"]:
                overlapping_masks = self.pred_logits[fr_idx]
                for tgt_id in fr_targets["overlap"]:
                    tgt_msk = overlapping_masks[tgt_id]
                    center_coords = get_center_coords(tgt_msk)
                    local_tgt_id = clip_orig_to_serial_id[tgt_id]
                    clip_fg_coords_list.append([center_coords[0], center_coords[1], local_tgt_id, local_fr_idx, t])
                    clip_num_clicks_per_target[local_fr_idx][local_tgt_id-1] += 1
                    clip_max_timestamps[local_fr_idx] += 1
                    self.overlap_coords_list[fr_idx].append([center_coords[0], center_coords[1], tgt_id, fr_idx, t])
                    t += 1

                    # random_coords = get_random_coords(tgt_msk, 5)
                    # for coords in random_coords:
                    #     clip_fg_coords_list.append([coords[0], coords[1], local_tgt_id, local_fr_idx, t])
                    #     clip_num_clicks_per_target[local_fr_idx][local_tgt_id-1] += 1
                    #     clip_max_timestamps[local_fr_idx] += 1
                    #     self.overlap_coords_list[fr_idx].append([coords[0], coords[1], tgt_id, fr_idx, t])
                    #     t += 1

                
                if self.save_vis:
                    self.save_overlapping_frame_w_clicks(fr_idx, clip_idx)
                
        clip_fg_coords_list = sorted(clip_fg_coords_list, key=lambda x:x[2])
        inputs = {
            "images": torch.as_tensor(self.images[indices], dtype=torch.uint8),
            "num_clicks_per_object": clip_num_clicks_per_target,
            "fg_coords_list": clip_fg_coords_list,
            "bg_coords_list": clip_bg_coords_list,
            "max_timestamp_list": clip_max_timestamps,
            "indices": indices,
            "orig_to_serial_id": clip_orig_to_serial_id,
            "serial_to_orig_id": clip_serial_to_orig_id,
            # extras
            "panoptic_masks": self.gt_masks[indices]
        }

        self.curr_clip_input = inputs
        if not last_clip:
            self.curr_overlapping_frames = _indices[-self.num_overlapping_frames:]
        else:
            self.curr_overlapping_frames = None
        return inputs
    
    
    def find_targets_in_clip(self, indices):
        """
        clip targets and where to find them
        """
        frames_to_sample = {}
        clip_target_ids = set()
        
        overlapping_frames = sorted(self.curr_overlapping_frames if self.curr_overlapping_frames else [])

        if self.num_overlapping_frames > 0:
        
            # there are 2 types of frames: 
            # 1. overlapping frames - these frames have already been seen. no new targets 
            # can be found. however, we'd like to use any g.t. clicks sampled on these 
            # frames. These targets are the `overlap_gt` targets. Rest of the targets in 
            # these frames are to be found from the already predicted mask.
            # 2. non-overlapping frames - these frames may have new targets and we want to
            # track them as the `new` targets
            new = []
            overlap_gt = []
            overlap_gt_flat = []
            for fr_idx in indices:
                overlap_gt.append([])
                for fg_click in self.gt_fg_coords_list[fr_idx]:
                    overlap_gt[-1].append(fg_click[2])
                overlap_gt_flat.extend(overlap_gt[-1])
                new.append(list(set(self.target_appearance.get(fr_idx, [])) - self.target_discovery))
            
            for i, fr_idx in enumerate(indices):
                overlapping_targets = self.prev_clip_output.get(fr_idx, [])
                overlap = [tgt_id for tgt_id in overlapping_targets if tgt_id not in overlap_gt_flat]
                frames_to_sample[fr_idx] = {"overlap_gt": overlap_gt[i], "overlap": overlap, "new": new[i]}
                clip_target_ids.update(overlap_gt[i] + overlap + new[i])

        else:
            frames_to_sample = defaultdict(dict)
            for fr_idx in indices:
                # no overlap, each clip is independent; so find new targets in each frame
                new_targets = list(set(np.unique(self.gt_masks[fr_idx])[1:]) - self.target_discovery)
                frames_to_sample[fr_idx]["new"] = new_targets
                clip_target_ids.update(new_targets)
                self.target_discovery.update(new_targets)
                frames_to_sample[fr_idx]["overlap_gt"] = []
                frames_to_sample[fr_idx]["overlap"] = []

        if self.num_overlapping_frames > 0:
            # maintain target discovery for next clip
            self.target_discovery.update(clip_target_ids)
        else:
            # reset target discovery for next clip
            self.target_discovery = set()
        
        return clip_target_ids, frames_to_sample
    
    
    def get_serial_indices(self, indices):
        last_clip = False
        if indices[-1] == "_":
            # last clip in forward/backward propagation
            indices = indices[:-1]
            last_clip = True
        if len(indices) >= 2 and indices[1] < indices[0]:
            indices = indices[::-1]
            if self.refine_frame:
                self.curr_overlapping_frames = self.refine_frame
                self.prev_clip_output = self.refine_frame
                self.refine_frame = None
        return indices, last_clip
    

    def record_gt_click(self, frame_idx: int, tgt_id: int, coords: List[int]) -> None:
        """
        Record a click in global buffers

        Args:
            frame_idx: int, the frame on which the click was sampled
            tgt_id: int, the ID of the target on which the click was sampled
            coords: list(int), the click location
        """
        if tgt_id == self.bg_id:
            self.gt_bg_coords_list[frame_idx].append(([coords[0], coords[1], -1, frame_idx, self.t]))
        else:
            self.gt_fg_coords_list[frame_idx].append([coords[0], coords[1], tgt_id, frame_idx, self.t])
            self.num_clicks_per_target[frame_idx][tgt_id-1] += 1
        
        self.num_clicks_per_frame[frame_idx] += 1
        self.max_timestamps[frame_idx] = self.t
        self.t+=1

        self.update_not_clicked_map(frame_idx, coords)
        
    
    def update_not_clicked_map(self, frame_idx: int, coords: List[int]) -> None:
        """
        Update the `not_clicked_map` with the sampled click to avoid sampling at or near this 
        click again. Strategy is specified by `self.sampling_strategy`
        
        0: new click avoids all the previously sampled click locations
        1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks

        Args:
            frame_idx: int, frame index
            coords: List[int], click coordinates in the format [y,x]
        """
        # update not_clicked_map with the sampled click location
        if self.sampling_strategy == 0:
            self.not_clicked_map[frame_idx][coords[0], coords[1]] = False
        elif self.sampling_strategy == 1:
            _pm = create_circular_mask(self.H, self.W, centers=[[coords[0], coords[1]]], radius=self.click_radius)
            self.not_clicked_map[frame_idx][np.where(_pm)] = False
        else:
            raise RuntimeError(f"Choose sampling strategy from: [0: only exclude click location, \
                1: exclude a circular area around click location specified by click_radius]")
        

    def store_prediction(self, binary_pred_masks: torch.Tensor, pred_logits: torch.Tensor, clip_idx:int) -> None:
        """
        Store predicted masks of a clip

        Args:
            binary_pred_masks: T,N,H,W np.ndarray predicted binary masks
            pred_logits: T,N,H,W np.ndarray predicted mask logits
        """
        confidence_threshold = 0.8
        
        indices = self.curr_clip_input["indices"]
        
        if self.curr_overlapping_frames:
            overlapping_frames = [indices.index(fr_idx) for fr_idx in self.curr_overlapping_frames]
            overlapping_pred_masks = binary_pred_masks[overlapping_frames]
            overlapping_pred_proba = pred_logits[overlapping_frames]
            
            # find overlapping targets from binary masks (0-indexed)
            overlapping_targets = overlapping_pred_masks.any(dim=(0, 2, 3)).nonzero(as_tuple=True)[0]
            # find frames where each target shows max confidence (0-indexed)
            max_response_frames = overlapping_pred_proba.amax(dim=(2, 3)).argmax(dim=0)
            # keep only the predicted targets
            max_response_frames = max_response_frames[overlapping_targets]
            
            # for each predicted target, store the original ID and also the corresponding 
            # channel index in the prediction logits
            frames_to_sample = defaultdict(list)
            for fr_idx, tgt_id in zip(max_response_frames, overlapping_targets):
                frames_to_sample[indices[overlapping_frames[fr_idx.item()]]].append(self.curr_clip_input["serial_to_orig_id"][tgt_id.item()+1])
            
            # frames_to_sample = defaultdict(list)
            # for fr_idx in overlapping_frames:
            #     pred_targets = binary_pred_masks[fr_idx].any(dim=(1,2)).nonzero(as_tuple=True)[0]
            #     frames_to_sample[indices[fr_idx]] = [self.curr_clip_input["serial_to_orig_id"][tgt_id.item()+1] for tgt_id in pred_targets]

            self.prev_clip_output = frames_to_sample
        
        # predicted panoptic maps
        T,N,H,W = binary_pred_masks.shape
        panoptic_pred_masks = np.full((T,H,W), fill_value=self.bg_id).astype(np.uint8)
        
        for fr_idx, global_fr_idx in enumerate(indices):
            fr_pred_logits = {}
            for tgt_id in range(N):
                orig_id = self.curr_clip_input["serial_to_orig_id"][tgt_id+1]
                mask = binary_pred_masks[fr_idx][tgt_id]
                if mask.any():
                    panoptic_pred_masks[fr_idx][np.where(mask==1)] = orig_id
                fr_pred_logits[orig_id] = (pred_logits[fr_idx][tgt_id] > confidence_threshold).numpy().astype(np.uint8)

            self.pred_logits[global_fr_idx] = fr_pred_logits
        self.pred_masks[indices] = panoptic_pred_masks

        if self.save_vis:
            self.save_visualization(indices, clip_idx)


    def save_visualization(self, indices: Optional[List[int]]=None, clip_idx: int=0, alpha: float=0.5) -> None:
        """
        Save predicted mask visualization to the disc

        Args:
            indices: Optional[List(int)], save visualizations for specified indices
            alpha: float, transparency to be applied to visualize mask overlaid on image
        """
        
        # visualization path for current round
        vis_path = os.path.join(self.path_to_visualization, f"round_{str(self.round_num)}")
        serial_mask_vis_path = os.path.join(vis_path, "masks")
        serial_overlaid_vis_path = os.path.join(vis_path, "overlaid")
        
        if not os.path.isdir(vis_path):
            os.makedirs(vis_path)
            os.makedirs(serial_mask_vis_path)
            os.makedirs(serial_overlaid_vis_path)

        
        if not indices: # display all frames
            indices = np.arange(self.T).tolist()
        save_masks = self.pred_masks[indices].copy()
        
        for fr_idx, fr_msk in zip(indices, save_masks):
            
            im = self.images[fr_idx].transpose(1,2,0)
            if self.resize:
                im = cv2.resize(im.copy(), (self.orig_W, self.orig_H))
                fr_msk = np.resize(fr_msk.copy(), (self.orig_H, self.orig_W))
                # TODO - scale clicks
            
            # save mask
            fr_msk = Image.fromarray(fr_msk.astype(np.uint8))
            fr_msk.putpalette(color_map)
            fr_msk.save(os.path.join(serial_mask_vis_path, f"clip_{clip_idx}_{fr_idx}.png"))
            
            # save mask overlaid on image with clicks
            image_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            fr_msk = np.array(fr_msk.convert("RGB"))
            mask_bgr = cv2.cvtColor(fr_msk, cv2.COLOR_RGB2BGR)
            overlaid = cv2.addWeighted(image_bgr, 1 - alpha, mask_bgr, alpha, 0)
            if len(self.gt_bg_coords_list[fr_idx]) > 0:
                show_points(overlaid, self.gt_bg_coords_list[fr_idx], 0)
            if len(self.gt_fg_coords_list[fr_idx]) > 0:
                show_points(overlaid, self.gt_fg_coords_list[fr_idx], 1)
            cv2.imwrite(os.path.join(serial_overlaid_vis_path, f"clip_{clip_idx}_{fr_idx}.png"), overlaid)
    
    
    def save_overlapping_frame_w_clicks(self, frame_idx: int, clip_idx: int, alpha: float=0.5) -> None:
        
        vis_path = os.path.join(self.path_to_visualization, f"round_{str(self.round_num)}")
        clip_vis_path = os.path.join(vis_path, "overlapping_clicks", str(clip_idx))
        os.makedirs(clip_vis_path, exist_ok=True)

        im = self.images[frame_idx].transpose(1,2,0).copy()
        fr_msk = self.pred_masks[frame_idx].copy()
        if self.resize:
            im = cv2.resize(im, (self.orig_W, self.orig_H))
            fr_msk = np.resize(fr_msk, (self.orig_H, self.orig_W))
            # TODO - scale clicks
        
        # save mask overlaid on image with clicks
        image_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        fr_msk = Image.fromarray(fr_msk.astype(np.uint8))
        fr_msk.putpalette(color_map)
        fr_msk = np.array(fr_msk.convert("RGB"))
        mask_bgr = cv2.cvtColor(fr_msk, cv2.COLOR_RGB2BGR)
        overlaid = cv2.addWeighted(image_bgr, 1 - alpha, mask_bgr, alpha, 0)
        show_points(overlaid, self.overlap_coords_list[frame_idx], 2)
        cv2.imwrite(os.path.join(clip_vis_path, f"clip_{clip_idx}_{frame_idx}.png"), overlaid)
    
    
    def get_corrective_click(self, frame_idx: int, tgt_id: int) -> int:
        """
        Obtain a corrective click on the specified target in the specified frame

        Args:
            frame_idx: int, frame to sample the click on
            tgt_id: int, ID of the target to sample the click on
        
        Returns:
            tgt_index: int, ID of the target present at the sampled click location in
                the g.t. mask of the frame `frame_idx`
        """
        gt_instance_mask = np.asarray(self.gt_masks[frame_idx] == tgt_id, dtype=np.bool_)
        pred_instance_mask = np.asarray(self.pred_masks[frame_idx] == tgt_id, dtype=np.bool_)
        np.save("/home/roy/REPOS/dynamite_video/debug/important/refinement_target_selection/gt_instance_mask.npy", gt_instance_mask)
        np.save("/home/roy/REPOS/dynamite_video/debug/important/refinement_target_selection/pred_instance_mask.npy", pred_instance_mask)

        # false negative map - g.t. foreground missed by the prediction
        fn_mask = np.logical_and(gt_instance_mask, np.logical_not(pred_instance_mask))
        # false positive map - wrongly predicted area outside object boundary
        fp_mask = np.logical_and(np.logical_not(gt_instance_mask), pred_instance_mask)

        # distance transform to find the center of the error region
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]
        
        # avoid regions around already sampled clicks
        fn_mask_dt = fn_mask_dt * self.not_clicked_map[frame_idx]
        fp_mask_dt = fp_mask_dt * self.not_clicked_map[frame_idx]

        # choose the bigger error region
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        is_positive = fn_max_dist > fp_max_dist

        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        t = len(coords_y) // 2
        sample_locations = [coords_y[t], coords_x[t]]
        tgt_index = self.gt_masks[frame_idx][coords_y[t], coords_x[t]]

        if self.save_vis:
            vis_path = os.path.join(self.path_to_visualization, f"round_{str(self.round_num)}/corrections")
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            
            image_bgr = cv2.cvtColor(self.images[frame_idx].transpose(1,2,0), cv2.COLOR_RGB2BGR)
            fr_gt_msk = Image.fromarray(self.gt_masks[frame_idx].astype(np.uint8))
            fr_gt_msk.putpalette(color_map)
            fr_gt_msk = cv2.cvtColor(np.array(fr_gt_msk.convert("RGB")), cv2.COLOR_RGB2BGR)
            overlaid_gt = cv2.addWeighted(image_bgr, 0.5, fr_gt_msk, 0.5, 0)
            show_points(overlaid_gt, [[sample_locations[0], sample_locations[1], 0,0,0]], 2)
            cv2.imwrite(os.path.join(vis_path, f"correction_click_gt_fr_{frame_idx}_tgt_{tgt_id}.png"), overlaid_gt)
            
            fr_pred_mask = Image.fromarray(self.pred_masks[frame_idx].astype(np.uint8))
            fr_pred_mask.putpalette(color_map)
            fr_pred_mask = cv2.cvtColor(np.array(fr_pred_mask.convert("RGB")), cv2.COLOR_RGB2BGR)
            overlaid_pred = cv2.addWeighted(image_bgr, 0.5, fr_pred_mask, 0.5, 0)
            show_points(overlaid_pred, [[sample_locations[0], sample_locations[1], 0,0,0]], 2)
            cv2.imwrite(os.path.join(vis_path, f"correction_click_pred_fr_{frame_idx}_tgt_{tgt_id}.png"), overlaid_pred)
        
        self.record_gt_click(frame_idx, tgt_index, sample_locations)

        # next round will start from frame `frame_idx`, so sample clicks from the prediction
        overlapping_targets = np.unique(self.pred_masks[frame_idx])    # includes bg
        self.curr_overlapping_frames = [frame_idx]
        self.prev_clip_output = {frame_idx: [tgt_id for tgt_id in overlapping_targets if tgt_id!=self.bg_id]}
        self.refine_frame = self.prev_clip_output
        
        return tgt_index