import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F

from collections import defaultdict
from PIL import Image
from typing import List, Mapping, Tuple

from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.evaluation.eval_utils import *
from dynamite_video.evaluation.metrics.metrics import compute_stq

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
            save_vis: bool,
            min_mask_area: int,
            connected_component_sampling: bool,
            debug: bool=False,   # TODO
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
        
        # serialize target IDs, 1-indexed
        self.orig_to_serial_ids, self.serial_to_orig_ids = serialize_target_ids(sorted(self.sequence.object_ids))
        # serial target ids
        self.target_ids = sorted(self.serial_to_orig_ids.keys())
        # bg label
        self.bg_id = 0
        
        # load tensors
        
        # load images, T,orig_H,orig_W,3
        self.images = self.sequence.load_images()
        # load ground truth panoptic masks, T,orig_H,orig_W with serialized IDs
        self.gt_masks, _, self.target_appearance = self.sequence.prepare_eval_masks(self.orig_to_serial_ids, self.bg_id)
        self.bg_masks = (self.gt_masks==self.bg_id)
        # apply transformation
        self.input_H, self.input_W = self.compute_tfm_sizes(tfms)

        # click level information
        
        # click radius - region around an existing click that is excluded when sampling a new click
        self.click_radius = 5
        # Strategy to avoid regions while sampling next clicks
        # 0: new click avoids all the previously sampled click locations
        # 1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks
        self.sampling_strategy = 1
        # sample a click only if the mask is bigger than a threshold
        self.min_mask_area = min_mask_area
        # whether to sample a single click from object center or from all connected components
        self.connected_component_sampling = connected_component_sampling
        # maintain a map of regions to avoid during click sampling, initially all true
        self.not_clicked_map = np.ones_like(self.gt_masks).astype(np.bool_)
        # num clicks on each target in each frame
        self.num_clicks_per_frame = np.zeros((self.T), dtype=np.uint16)
        # num clicks on each target in each frame
        self.num_clicks_per_target = np.zeros((self.T, self.N), dtype=np.uint16)
        # foreground clicks sampled on each target in each frame from the ground truth mask
        self.gt_fg_coords_list = [[] for _ in range(self.T)]
        # background clicks sampled on each frame from the ground truth mask
        self.gt_bg_coords_list = [[] for _ in range(self.T)]
        # foreground clicks sampled on each target from the predicted mask of some overlapping frame
        self.overlap_coords_list = [[] for _ in range(self.T)]
        
        # prediction information
        
        # to store predicted masks
        self.pred_masks = np.zeros((self.T, self.orig_H, self.orig_W), dtype=np.uint8)
        # to store prediction logits
        self.pred_logits = [[] for _ in range(self.T)]
        # I/O
        self.curr_clip_input = {}
        self.prev_clip_output = {}
        self.curr_overlapping_frames = None
        self.refine_frame = None

        # rounding info
        self.round_num = 0
        # length of clips to be extracted from the sequence
        self.clip_length = dataset_meta["clip_length"]
        # overlap between successive clips
        self.num_overlapping_frames = dataset_meta["num_overlapping_frames"]
        
        # visualization paths
        self.output_dir = output_dir
        self.save_vis = save_vis
        self.debug = debug
        if self.save_vis:
            # path to save predicted masks
            self.path_to_visualization = os.path.join(output_dir, "masks", self.sequence.id)
            os.makedirs(self.path_to_visualization)
            if self.debug:
                self.path_to_debug_visualization = os.path.join(output_dir, "debug", self.sequence.id)
                os.makedirs(self.path_to_debug_visualization)
                # category labels
                category_labels = get_category_labels(self.orig_to_serial_ids, dataset_meta)
                save_color_palette(category_labels, self.path_to_debug_visualization)

    
    def set_budget(self, max_interactions: int):
        self.budget = np.full(self.N, max_interactions)
        return max_interactions * self.N
        

    def compute_tfm_sizes(self, tfms: Mapping) -> Tuple[int, int]:
        """
        Compute transformed resolution

        Args:
            tfms: mapping containing transformation info from config, cfg.INPUT
        
        Returns:
            input_H, input_W: tuple, transformed height and width
        """
        input_H, input_W = self.orig_H, self.orig_W
        self.resize = False
        self.scale_H = 1.0
        self.scale_W = 1.0
        if tfms.AUGMENTATION.RESIZE_TEST:
            input_H, input_W = compute_resized_dims(self.orig_H, self.orig_W,
                                                        min_dim=tfms.AUGMENTATION.MIN_DIM_TEST,
                                                        max_dim=tfms.AUGMENTATION.MAX_DIM_TEST,
                                                    )
            #self.images = resize_images(self.images, input_H, input_W)
            #self.gt_masks = resize_masks(self.gt_masks, input_H, input_W, binary=False)
            #self.ignore_masks = resize_masks(self.ignore_masks, input_H, input_W, binary=False)
            self.resize=True
            self.scale_H = input_H / self.orig_H
            self.scale_W = input_W / self.orig_W

        self.images = np.transpose(self.images, (0, 3, 1, 2))   # T,3,H,W
        if tfms.RGB:
            self.images = np.flip(self.images, 1).copy()        # BGR -> RGB
        return input_H, input_W
        

    def generate_clip_indices(self, start: int) -> List[List[int]]:
        """
        Called at the beginning of each round. Given a start index, generate list 
        of indices of clips from the sequence. Also resets the clicks sampled on 
        overlapping frames.
        
        NOTE: framework only supports forward propagation
        ~~If the start index is in the middle of the sequence, it generates clips in \
            both forward and backward directions.~~

        Args:
            start: int, index of the first frame of the first clip
        
        Returns:
            indices: List[List[int]], list of frame indices per clip
        """
        step = self.clip_length - self.num_overlapping_frames
        clips = []

        # forward
        pos = start
        while pos < self.T:
            end = min(pos + self.clip_length, self.T)
            indices = list(range(pos, end))
            is_last = end == self.T
            clips.append({"indices": indices, "is_backward": False, "is_last": is_last})
            if is_last:
                break
            pos += step
        
        # NOTE: framework only supports forward propagation
        # backward
        # pos = start - 1 + self.num_overlapping_frames
        # while pos >= 0:
        #     end = max(pos - self.clip_length + 1, 0)
        #     indices = list(range(pos, end-1, -1))
        #     is_last = end == 0
        #     clips.append({"indices": indices, "is_backward": True, "is_last": is_last})
        #     if is_last:
        #         break
        #     pos -= step
        
        # resets overlapping coords at the beginning of each round
        self.overlap_coords_list = [[] for _ in range(self.T)]
        return clips

    
    def extract_clip(self, _indices: List[int]) -> Mapping:
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

        indices, _, is_last = self.get_serial_indices(_indices)

        # ground truth target objects
        # gt_panoptic_masks = self.gt_masks[indices]
        
        # find targets in the clip
        clip_target_ids, frames_to_sample = self.find_targets_in_clip(indices)
        # serialize target IDs
        clip_orig_to_serial_id, clip_serial_to_orig_id = serialize_target_ids(clip_target_ids)

        # click info must be re-adjusted to be consistent with clip level frame indices and target ids
        clip_fg_coords_list, clip_bg_coords_list = [], []
        clip_num_clicks_per_target = np.zeros((len(indices), len(clip_target_ids)), dtype=np.uint16)
        clip_max_timestamps = [0 for _ in indices]
        t = 1
        
        for global_fr_idx, fr_targets in frames_to_sample.items():
            local_fr_idx = indices.index(global_fr_idx)
            
            # already sampled g.t. clicks
            for fg_click in self.gt_fg_coords_list[global_fr_idx]:
                serial_id = clip_orig_to_serial_id[fg_click[2]]
                clip_fg_coords_list.append([fg_click[0]*self.scale_H, fg_click[1]*self.scale_W, serial_id, local_fr_idx, t])
                clip_num_clicks_per_target[local_fr_idx][serial_id-1] += 1
                clip_max_timestamps[local_fr_idx] = t
                t += 1
            for bg_click in self.gt_bg_coords_list[global_fr_idx]:
                clip_bg_coords_list.append([bg_click[0]*self.scale_H, bg_click[1]*self.scale_W, bg_click[2], local_fr_idx, t])
                clip_max_timestamps[local_fr_idx] = t
                t += 1
            
            # new targets
            for orig_id in fr_targets["new"]:
                serial_id = clip_orig_to_serial_id[orig_id]
                mask = (self.gt_masks[global_fr_idx] == orig_id).astype(np.uint8) * self.not_clicked_map[global_fr_idx]
                center_coords = get_component_center_coords(mask, 
                                                            cc=self.connected_component_sampling,
                                                            budget=self.budget[orig_id-1],
                                                            min_area=self.min_mask_area
                                                        )
                for cc in center_coords:
                    clip_fg_coords_list.append([cc[0]*self.scale_H, cc[1]*self.scale_W, serial_id, local_fr_idx, t])
                    self.record_gt_click(global_fr_idx, orig_id, cc)
                    t += 1
                clip_num_clicks_per_target[local_fr_idx][serial_id-1] += len(center_coords)
                clip_max_timestamps[local_fr_idx] = t-1
                
            
            # targets in overlapping frames
            if fr_targets["overlap"]:
                overlapping_masks = self.pred_logits[global_fr_idx]
                for orig_id in fr_targets["overlap"]:
                    serial_id = clip_orig_to_serial_id[orig_id]
                    center_coords = get_component_center_coords(overlapping_masks[orig_id], 
                                                                cc=self.connected_component_sampling,
                                                                budget=None,
                                                                min_area=self.min_mask_area
                                                            )
                    for cc in center_coords:
                        clip_fg_coords_list.append([cc[0]*self.scale_H, cc[1]*self.scale_W, serial_id, local_fr_idx, t])
                        self.overlap_coords_list[global_fr_idx].append([cc[0], cc[1], orig_id])
                        t += 1
                    clip_num_clicks_per_target[local_fr_idx][serial_id-1] += len(center_coords)
                    clip_max_timestamps[local_fr_idx] = t-1
                
        clip_fg_coords_list = sorted(clip_fg_coords_list, key=lambda x:x[2])

        # scale to resized shape
        clip_images = torch.as_tensor(self.images[indices],dtype=torch.uint8)
        if self.resize:
            clip_images = F.interpolate(clip_images, (self.input_H, self.input_W), mode='bilinear', align_corners=False)

        inputs = {
            "images": clip_images,
            "num_clicks_per_object": clip_num_clicks_per_target,
            "fg_coords_list": clip_fg_coords_list,
            "bg_coords_list": clip_bg_coords_list,
            "max_timestamp_list": clip_max_timestamps,
            "indices": indices,
            "orig_to_serial_id": clip_orig_to_serial_id,
            "serial_to_orig_id": clip_serial_to_orig_id,
            # extras
            #"panoptic_masks": gt_panoptic_masks,
            "orig_res": [self.orig_H, self.orig_W],
            "overlapping_frames": self.curr_overlapping_frames,
            "overlapping_masks": self.pred_masks[self.curr_overlapping_frames] if self.curr_overlapping_frames is not None else None,
        }

        # record the current clip, and the overlapping frame indices for the next clip
        self.curr_clip_input = inputs
        if not is_last:
            self.curr_overlapping_frames = _indices["indices"][-self.num_overlapping_frames:]
        else:
            self.curr_overlapping_frames = None
        return inputs
    
    
    def find_targets_in_clip(self, indices):
        """
        clip targets and where to find them
        """
        frames_to_sample = {}
        clip_target_ids = set()

        if self.num_overlapping_frames > 0:
        
            # there are 2 types of frames in each input clip: 
            # 1. Overlapping frames - already seen - no new targets can be found. but, we'd 
            # like to use any g.t. clicks sampled on these frames. 
            # Additionally, we sample a click on every target predicted in these overlapping frames
            # 2. Non-overlapping frames - may have new targets we want to track
            
            # if there's g.t. clicks already sampled in these frames
            gt_click_targets = set()
            for fr_idx in indices:
                for (_,_,tgt_id) in self.gt_fg_coords_list[fr_idx]:
                        gt_click_targets.add(tgt_id)
            clip_target_ids.update(gt_click_targets)
            
            for fr_idx in indices:
                # new targets = any target that has not been discovered yet
                new_targets = self.target_appearance.pop(fr_idx, [])
                # overlapping targets = any target appearing in the predicted masks of overlapping frames
                overlapping_targets = self.prev_clip_output.get(fr_idx, [])
                
                frames_to_sample[fr_idx] = {"overlap": overlapping_targets, "new": new_targets}
                clip_target_ids.update(overlapping_targets + new_targets)

        else:
            # no overlap, each clip is independent; so find new targets in each frame
            targets_discovered = set()
            for fr_idx in indices:
                new_targets = list(set(np.unique(self.gt_masks[fr_idx])[1:]) - targets_discovered)
                targets_discovered.update(new_targets)
                frames_to_sample[fr_idx] = {"overlap": [], "new": new_targets}
                clip_target_ids.update(new_targets)
        
        return clip_target_ids, frames_to_sample
    
    
    def get_serial_indices(self, indices):
        if indices["is_backward"]:
            _indices = indices["indices"][::-1]
        else:
            _indices = indices["indices"]
        return _indices, indices["is_backward"], indices["is_last"]
    

    def record_gt_click(self, frame_idx: int, tgt_id: int, coords: List[int], bg_true_id: int=None) -> None:
        """
        Record a click in global buffers

        Args:
            frame_idx: int, the frame on which the click was sampled
            tgt_id: int, the ID of the target on which the click was sampled
            coords: list(int), the click location
            bg_true_id: int, if a click is sampled on BG, the click counts towards
                    the object it refines
        """
        if tgt_id == self.bg_id:
            self.gt_bg_coords_list[frame_idx].append([coords[0], coords[1], -1])
            self.num_clicks_per_target[frame_idx][bg_true_id-1] += 1
            self.budget[bg_true_id-1] -= 1
        else:
            self.gt_fg_coords_list[frame_idx].append([coords[0], coords[1], tgt_id])
            self.num_clicks_per_target[frame_idx][tgt_id-1] += 1
            self.budget[tgt_id-1] -= 1
        
        self.num_clicks_per_frame[frame_idx] += 1
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
            _pm = create_circular_mask(self.orig_H, self.orig_W, centers=[[coords[0], coords[1]]], radius=self.click_radius)
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
        # upsample to orig resolution
        binary_pred_masks = F.interpolate(binary_pred_masks.float(), size=(self.orig_H, self.orig_W),mode="nearest").to(binary_pred_masks.dtype)
        pred_logits = F.interpolate(pred_logits.float(), size=(self.orig_H, self.orig_W),mode="bilinear").to(pred_logits.dtype)

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

            self.prev_clip_output = frames_to_sample
        
        # predicted panoptic maps
        T,N,H,W = binary_pred_masks.shape
        pred_panoptic_masks = np.full((T,H,W), fill_value=self.bg_id).astype(np.uint8)
        
        for fr_idx, global_fr_idx in enumerate(indices):
            fr_pred_logits = {}
            for tgt_id in range(N):
                orig_id = self.curr_clip_input["serial_to_orig_id"][tgt_id+1]
                mask = binary_pred_masks[fr_idx][tgt_id]
                if mask.any():
                    pred_panoptic_masks[fr_idx][np.where(mask==1)] = orig_id
                fr_pred_logits[orig_id] = (pred_logits[fr_idx][tgt_id] > confidence_threshold).numpy().astype(np.uint8)

            self.pred_logits[global_fr_idx] = fr_pred_logits
        self.pred_masks[indices] = pred_panoptic_masks

        if self.save_vis:
            self.save_visualization(clip_idx)
    
    
    def find_refinement_targets(self, target_level_scores: Mapping, iou_threshold: float, eval_strategy: str):
        """
        Poorly-segmented objects and where to find them.

        Args:
            * target_level_scores: object level SQ scores
            * iou_threshold: float, IoU threshold
            * eval_strategy: "worst" to select only the worst candidate, 
                             "random" to select one candidate at random, 
                             "all" to select all candidates under `iou_threshold`
        """
        # select candidates for refinement
        valid_idx = np.where(target_level_scores["sq_per_target"] < iou_threshold)[0]
        all_candidates = valid_idx[np.argsort(target_level_scores["sq_per_target"][valid_idx])]
        if eval_strategy == "random":
            np.random.shuffle(all_candidates)

        refinements = []
        for tgt_id in all_candidates:
            # find the best frame to sample a click on
            refine_frame = self.find_refinement_frame(tgt_id, target_level_scores["sq_per_frame_per_target"], iou_threshold)
            refined_tgt_id = self.get_corrective_click(frame_idx=refine_frame, refine_tgt_id=tgt_id+1)
            if refined_tgt_id is None:
                continue
            # print(f"Sampled a click on {refined_tgt_id} (originally, {tgt_id+1}) at frame {refine_frame}")
            refinements.append({"Target": tgt_id+1, "Frame": refine_frame, "GT objects clicked": refined_tgt_id})
            if eval_strategy in ["worst", "random"]:
                break
        return refinements


    def find_refinement_frame(self, tgt_id, sq_per_frame_per_target, iou_threshold):
        
        candidate_frames = sq_per_frame_per_target[:, tgt_id] < iou_threshold

        # frame intervals where the target's IoU goes below threshold
        diff = np.diff(candidate_frames.astype(int))
        starts = np.where(diff == 1)[0] + 1
        if candidate_frames[0]:
            starts = np.r_[0, starts]
        ends = np.where(diff == -1)[0]
        if candidate_frames[-1]:
            ends = np.r_[ends, len(candidate_frames) - 1]
        weak_intervals = list(zip(starts, ends))

        if len(weak_intervals) == 0:
            raise RuntimeError(f"No weak frame intervals found for an object to be refined!")
        
        if len(weak_intervals) == 1:
            return weak_intervals[0][0]
        
        # severity scores
        best_score = -float("inf")
        best_start = None
        # bias toward choosing earlier frames
        alpha = 0.5 * (1 - iou_threshold)
        for start, end in weak_intervals:
            # how bad the masks in the interval are
            mean_iou = sq_per_frame_per_target[start:end+1, tgt_id].mean()
            # how harmful the interval is
            severity = (end - start + 1) * (1.0 - mean_iou)
            # penalize later intervals
            score = severity - alpha * start

            if score > best_score:
                best_score = score
                best_start = start
        return best_start

    
    def get_corrective_click(self, frame_idx: int, refine_tgt_id: int) -> int:
        """
        Obtain a corrective click on the specified target in the specified frame

        Args:
            frame_idx: int, frame to sample the click on
            refine_tgt_id: int, ID of the target to sample the click on
        
        Returns:
            gt_tgt_index: int, ID of the target present at the sampled click location in
                the g.t. mask of the frame `frame_idx`
        """
        gt_instance_mask = np.asarray(self.gt_masks[frame_idx] == refine_tgt_id, dtype=np.bool_)
        pred_instance_mask = np.asarray(self.pred_masks[frame_idx] == refine_tgt_id, dtype=np.bool_)

        # false negative map - g.t. foreground missed by the prediction
        fn_mask = np.logical_and(gt_instance_mask, np.logical_not(pred_instance_mask))
        # false positive map - wrongly predicted area outside object boundary
        fp_mask = np.logical_and(np.logical_not(gt_instance_mask), pred_instance_mask)

        # distance transform to find the center of the error region
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
        fn_mask = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fn_mask = fn_mask[1:-1, 1:-1]
        fp_mask = fp_mask[1:-1, 1:-1]
        
        # avoid regions around already sampled clicks
        fn_mask = fn_mask * self.not_clicked_map[frame_idx]
        fp_mask = fp_mask * self.not_clicked_map[frame_idx]

        # choose the bigger error region
        fn_max_dist = np.max(fn_mask)
        fp_max_dist = np.max(fp_mask)

        # sample the click at the center of the error region
        if fn_max_dist > fp_max_dist:
            # coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
            center_coords = get_component_center_coords(fn_mask, 
                                                        cc=True, #self.connected_component_sampling,
                                                        budget=None, 
                                                        min_area=self.min_mask_area*2)
        else:
            # coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]
            center_coords = get_component_center_coords(fp_mask, 
                                                        cc=True, #self.connected_component_sampling,
                                                        budget=None, 
                                                        min_area=self.min_mask_area*2)
        
        # t = len(coords_y) // 2
        # sample_locations = [coords_y[t], coords_x[t]]
        # gt_tgt_index = self.gt_masks[frame_idx][coords_y[t], coords_x[t]]
        
        # if the click is on a foreground object, check its remaining budget
        # if gt_tgt_index!= self.bg_id and self.budget[gt_tgt_index-1] <= 0:
        #     return None
        # self.record_gt_click(frame_idx, gt_tgt_index, sample_locations)
            
        accepted_tgts = []
        corrections = []
        for cc in center_coords:
            gt_tgt_index = self.gt_masks[frame_idx][tuple(cc)]
            if self.budget[refine_tgt_id-1] <= 0:
                break
            accepted_tgts.append(gt_tgt_index)
            corrections.append(tuple(cc))
            # the click should count towards the budget of the refine target
            self.record_gt_click(frame_idx, refine_tgt_id, tuple(cc))
        if len(accepted_tgts) == 0:
            return None

        # next round will start from frame `frame_idx`, so sample clicks from the prediction
        # overlapping_targets = np.unique(self.pred_masks[frame_idx])    # includes bg
        # self.curr_overlapping_frames = [frame_idx]
        # prev_clip_output_targets = [tgt_id for tgt_id in overlapping_targets if tgt_id!=self.bg_id]
        
        # self.refine_frame = {frame_idx: prev_clip_output_targets}
        # self.prev_clip_output = self.refine_frame
        
        if self.save_vis:
            # corrections = [sample_locations]
            vis_path = os.path.join(self.path_to_debug_visualization, f"round_{str(self.round_num)}_corrections")
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            
            image_bgr = cv2.cvtColor(self.images[frame_idx].transpose(1,2,0), cv2.COLOR_RGB2BGR)
            fr_gt_msk = Image.fromarray(self.gt_masks[frame_idx].astype(np.uint8))
            fr_gt_msk.putpalette(color_map)
            fr_gt_msk = cv2.cvtColor(np.array(fr_gt_msk.convert("RGB")), cv2.COLOR_RGB2BGR)
            overlaid_gt = cv2.addWeighted(image_bgr, 0.5, fr_gt_msk, 0.5, 0)
            show_points(overlaid_gt, corrections, 2)
            cv2.imwrite(os.path.join(vis_path, f"correction_click_gt_fr_{frame_idx}_tgt_{refine_tgt_id}.png"), overlaid_gt)
            
            fr_pred_mask = Image.fromarray(self.pred_masks[frame_idx].astype(np.uint8))
            fr_pred_mask.putpalette(color_map)
            fr_pred_mask = cv2.cvtColor(np.array(fr_pred_mask.convert("RGB")), cv2.COLOR_RGB2BGR)
            overlaid_pred = cv2.addWeighted(image_bgr, 0.5, fr_pred_mask, 0.5, 0)
            show_points(overlaid_pred, corrections, 2)
            cv2.imwrite(os.path.join(vis_path, f"correction_click_pred_fr_{frame_idx}_tgt_{refine_tgt_id}.png"), overlaid_pred)
        
        return accepted_tgts
    

    def save_visualization(self, clip_idx):
        # visualization path for current round
        vis_path = os.path.join(self.path_to_visualization, f"round_{str(self.round_num)}")
        if not os.path.isdir(vis_path):
            os.makedirs(vis_path)
        if self.debug:
            vis_path_debug = os.path.join(self.path_to_debug_visualization, f"round_{str(self.round_num)}")
            if not os.path.isdir(vis_path_debug):
                os.makedirs(vis_path_debug)
            
        
        # input clip
        inputs = self.curr_clip_input
        overlapping_frames = inputs["overlapping_frames"]
        overlapping_masks = inputs["overlapping_masks"]

        for fr_idx in inputs["indices"]:
            
            im = self.images[fr_idx].transpose(1,2,0)
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            def to_bgr_im(arr):
                arr = Image.fromarray(arr.astype(np.uint8))
                arr.putpalette(color_map)
                arr_rgb = np.array(arr.convert("RGB"))
                arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
                return arr_bgr
            
            # current prediction)
            pred_bgr = to_bgr_im(self.pred_masks[fr_idx])
            
            if self.debug:
                # ground truth mask
                gt_bgr = to_bgr_im(self.gt_masks[fr_idx])
                # clicks sampled on ground truth mask
                if len(self.gt_bg_coords_list[fr_idx]) > 0:
                    show_points(gt_bgr, self.gt_bg_coords_list[fr_idx], 0)
                if len(self.gt_fg_coords_list[fr_idx]) > 0:
                    show_points(gt_bgr, self.gt_fg_coords_list[fr_idx], 1)
                combined = [gt_bgr]
                labels = ["Ground Truth"]

                # overlapping mask prediction from previous clip
                if overlapping_frames is not None and fr_idx in overlapping_frames:
                    overlap_bgr = to_bgr_im(overlapping_masks[overlapping_frames.index(fr_idx)])
                    if len(self.overlap_coords_list[fr_idx]) > 0:
                        show_points(overlap_bgr, self.overlap_coords_list[fr_idx], 2)
                    combined.append(overlap_bgr)
                    labels.append("Overlapping prediction")
                    
                combined.append(pred_bgr)
                labels.append("Prediction")

                # Stack vertically
                combined = cv2.vconcat(combined)
                h, w, _ = im_bgr.shape
                x = 10
                for i, label in enumerate(labels):
                    y = i * h + 30
                    # Get text size
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                    overlay = combined.copy()
                    cv2.rectangle(overlay, (x-5, y-th-5), (x+tw+5, y+5), (0, 0, 0), -1)
                    alpha = 0.6
                    combined = cv2.addWeighted(overlay, alpha, combined, 1 - alpha, 0)
                    cv2.putText(combined, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imwrite(f"{vis_path_debug}/clip_{clip_idx}_{fr_idx:04d}.png", combined)
            
            alpha = 0.5
            overlaid = cv2.addWeighted(im_bgr, 1 - alpha, pred_bgr, alpha, 0)
            if len(self.gt_bg_coords_list[fr_idx]) > 0:
                show_points(overlaid, self.gt_bg_coords_list[fr_idx], 0)
            if len(self.gt_fg_coords_list[fr_idx]) > 0:
                show_points(overlaid, self.gt_fg_coords_list[fr_idx], 1)
            cv2.imwrite(f"{vis_path}/clip_{clip_idx}_{fr_idx:04d}.png", overlaid)


    def calculate_score(self) -> Tuple[Mapping, float]:
        result = compute_stq(y_true=self.gt_masks, 
                            y_pred=self.pred_masks, 
                            target_ids=self.target_ids,
                            ignore_label=self.bg_id)
        
        scores = {
            "Round": self.round_num, 
            "#frames": self.T, 
            "#targets": self.N, 
            "#clicks": int(self.num_clicks_per_frame.sum()), 
            "STQ": result["STQ"], 
            "AQ": result["AQ"], 
            "SQ": result["SQ"],
        }
        target_level_scores = {
            "sq_per_target": result["sq_per_target"],
            "sq_per_frame_per_target": result["sq_per_frame_per_target"],
         #   "error_per_frame_per_target": result["error_per_frame_per_target"]
        }
        return scores, target_level_scores


    