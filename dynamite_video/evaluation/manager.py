import cv2
import numpy as np
import os
import torch

from PIL import Image

from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.evaluation.eval_utils import create_circular_mask, color_map, show_points, get_center_coords, serialize_object_ids

class SequenceManager:
    """
    Sequence manager
    """

    def __init__(self, sequence, dataset_meta, tfms):
        """
        Initialize with information on the sequence data, including ground truth masks

        Sequence Manager maintains all target-related info.

        Args:
            sequence: `GenericVideoSequence` instance of current video
            dataset_meta: dict with dataset level info on following properties:
                * category_labels: mapping between category ID and label, e.g., 0: 'road'
                * clip_length: int, length of each sub-sequence
                * num_overlapping_frames: int, overlap between consecutive sub-sequences
                * fps: int, sequence FPS
                * split: str, dataset split (val/test)
                * num_classes: int, number of semantic classes
                * things_list: list(int), semantic labels of `thing` classes
                * ignore_class: int, semantic label of `VOID` class
                * max_instances_per_category: int, max #instances to expect per semantic class
            tfms: transformation info from config, cfg.INPUT
        """

        # dataset level information
        
        # fps of dataset sequences
        self.fps = dataset_meta["fps"]
        # dataset split
        self.split = dataset_meta["split"]
        # num of semantic classes - minus 'void'
        self.num_classes = dataset_meta["num_classes"]
        # original class label to category name mapping
        self.category_labels = dataset_meta["category_labels"]
        # 'things' class labels
        self.things_list = dataset_meta["things_list"]
        # ignore class label
        self.ignore_class = dataset_meta["ignore_class"]
        # max num of instances per category
        self.max_instances_per_category = dataset_meta["max_instances_per_category"]
        
        # sequence level information
        
        # `GenericVideoSequence` instance of current video
        self.sequence = sequence
        # num of frames in the sequence
        self.T = len(self.sequence)
        # num of targets in the sequence (does not include `VOID` and `thing` classes)
        self.N = len(self.sequence.object_ids)
        # original spatial resolution
        self.orig_H, self.orig_W = self.sequence.image_dims

        # object level information
        
        self.orig_object_ids = self.sequence.object_ids
        # serialize object IDs, 1-indexed
        self.orig_to_serial_ids, self.serial_to_orig_ids = serialize_object_ids(sorted(self.orig_object_ids))
        # serial object ids
        self.object_ids = sorted(self.serial_to_orig_ids.keys())
        # bg label
        self.bg_id = 0
        # maintain a record to keep track of objects that were already discovered
        self.object_discovery = set()
        
        # load tensors
        
        # load images, T,orig_H,orig_W,3
        self.images = self.sequence.load_images()
        # load ground truth panoptic masks, T,orig_H,orig_W with serialized IDs
        self.gt_masks, self.ignore_masks, self.object_appearance = self.sequence.prepare_eval_masks(self.orig_to_serial_ids, self.bg_id)
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
        # allow sampling a click if object area is larger than a threshold
        self.MIN_MASK_AREA = 0
        # maintain a map of regions to avoid during click sampling, initially all true
        self.not_clicked_map = np.ones_like(self.gt_masks).astype(np.bool_)
        # foreground clicks sampled on each object in each frame
        self.fg_coords_list = [[] for _ in range(self.T)]
        # background clicks sampled on each frame
        self.bg_coords_list = [[] for _ in range(self.T)]
        # foreground clicks sampled on each object in each frame
        self.all_fg_clicks = [[] for _ in range(self.T)]
        # background clicks sampled on each frame
        self.all_bg_clicks = [[] for _ in range(self.T)]
        # num clicks on each object in each frame
        self.num_clicks_per_object = np.zeros((self.T, self.N), dtype=np.uint16)
        # num clicks per frame
        self.num_clicks_per_frame = np.zeros((self.T), dtype=np.uint16)
        # time stamp of the latest click on each frame
        self.max_timestamps = np.zeros((self.T), dtype=np.uint16)
        # first click at timestamp 1
        self.t = 1
        
        # prediction information
        
        # to store predicted masks
        self.pred_masks = np.zeros((self.T, self.H, self.W), dtype=np.uint8)
        # store IoU
        self.ious = np.full((self.T, self.N), fill_value=-1., dtype=np.single)
        self.frame_level_ious = np.zeros((self.T), dtype=np.single)

        # rounding info
        self.round_num = 0

        # length of clips to be extracted from the sequence
        self.clip_length = dataset_meta["clip_length"]
        # overlap between successive clips
        self.num_overlapping_frames = dataset_meta["num_overlapping_frames"]
        self.vis_path = dataset_meta["vis_path"]

        # overlapping queries
        self.prev_clip_input = {}
        self.prev_clip_output = {}
        

    def compute_tfm_sizes(self, tfms):
        """
        Compute transformed resolution

        tfms: transformation info from config, cfg.INPUT
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
        

    def generate_clip_indices(self, start):
        """
        Given a start index, generate list of indices of clips from the sequence.
        If the start index is in the middle of the sequence, it generates clips in
        both forward and backward directions.

        Args:
            start: int, index of the first frame of the first clip
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

        # generate clips that go back in time
        if start_copy>0:
            bwd = []
            start_copy = min(start_copy+self.num_overlapping_frames-1, self.T-1)
            while start_copy - self.clip_length >= 0:
                bwd.append(list(range(start_copy, start_copy-self.clip_length, -1)))
                start_copy -= step
            if len(bwd) > 0 and bwd[-1][-1] != 0:
                bwd.append(list(range(bwd[-1][-1] + self.num_overlapping_frames-1, -1, -1)))
            elif len(bwd) == 0:
                bwd.append(list(range(start_copy, -1, -1)))
            indices.extend(bwd)
        return indices

    
    def extract_clip(self, _indices):
        """
        Idea - 

        Case 1: First clip

        Sample a click on each target on the frame where it first appears

        Case 2: Intermediate clips

        From previous clip's predicted  binary masks, find the targets present
        For each predicted target, sample a clip from the high confidence region
        of the predicted mask logit
        """

        indices = _indices
        if len(indices) >= 2 and indices[1] < indices[0]:
            indices = _indices[::-1]

        # if there are any overlapping frames, find which targets to reconsider in current clip
        overlapping_frames = self.prev_clip_output.get("frames", None)

        # targets in the clip and where to find them
        frames_to_sample = {}
        clip_target_ids = []
        
        for fr_idx in indices:
            frames_to_sample[fr_idx] = {"new": [], "overlap": []}
            
            if self.num_overlapping_frames > 0:
                # look for new targets in the frame
                new_targets = list(set(self.object_appearance.get(fr_idx, [])) - self.object_discovery)
                frames_to_sample[fr_idx]["new"] = new_targets
                clip_target_ids.extend(new_targets)

                if overlapping_frames:
                    # look for overlapping targetss in the frame
                    overlapping_targets = overlapping_frames.get(fr_idx, [])
                    frames_to_sample[fr_idx]["overlap"] = overlapping_targets
                    clip_target_ids.extend(overlapping_targets)
            
            else:
                # each clip is independent, so find new objects in each frame
                new_targets = list(set(np.unique(self.gt_masks[fr_idx])[1:]) - self.object_discovery)
                frames_to_sample[fr_idx]["new"] = new_targets
                clip_target_ids.extend(new_targets)    
                self.object_discovery.update(new_targets)

        if self.num_overlapping_frames > 0:
            self.object_discovery.update(clip_target_ids)
        else:
            self.object_discovery = set()
        
        # serialize target IDs
        clip_orig_to_serial_id, clip_serial_to_orig_id = serialize_object_ids(clip_target_ids)

        # click info must be readjusted to be consistent with clip level frame indices and target ids
        clip_fg_coords_list, clip_bg_coords_list = [], []
        clip_num_clicks_per_target = np.zeros((len(indices), len(clip_target_ids)), dtype=np.uint16)
        clip_max_timestamps = [0 for _ in indices]
        
        for fr_idx, fr_targets in frames_to_sample.items():
            local_fr_idx = indices.index(fr_idx)

            for tgt_id in fr_targets["new"]:
                # sample a click on the g.t. mask of the new target
                tgt_mask = (self.gt_masks[fr_idx] == tgt_id).astype(np.uint8)
                center_coords = get_center_coords(tgt_mask * self.not_clicked_map[fr_idx])
                
                # record the click as [y,x,i,f,t]
                local_obj_id = clip_orig_to_serial_id[tgt_id]
                clip_fg_coords_list.append([
                    center_coords[0], center_coords[1], local_obj_id, local_fr_idx, self.t
                ])
                clip_num_clicks_per_target[local_fr_idx][local_obj_id-1] += 1
                clip_max_timestamps[local_fr_idx] += 1
                self.record_click(fr_idx, tgt_id, center_coords)

            if fr_targets["overlap"]:
                overlapping_masks = self.prev_clip_output["masks"][fr_idx]
                t = self.t
                for tgt_id, tgt_msk in zip(fr_targets["overlap"], overlapping_masks):
                    tgt_msk = tgt_msk.numpy().astype(np.uint8)
                    center_coords = get_center_coords(tgt_msk)
                    
                    local_obj_id = clip_orig_to_serial_id[tgt_id]
                    clip_fg_coords_list.append([
                        center_coords[0], center_coords[1], local_obj_id, local_fr_idx, t+1
                    ])
                    clip_num_clicks_per_target[local_fr_idx][local_obj_id-1] += 1
                    clip_max_timestamps[local_fr_idx] += 1
                    t += 1
                    if tgt_id == self.bg_id:
                        self.all_bg_clicks[fr_idx].append(([center_coords[0], center_coords[1], -1, fr_idx, t-1]))
                    else:
                        self.all_fg_clicks[fr_idx].append([center_coords[0], center_coords[1], tgt_id, fr_idx, t-1])
                
        inputs = {
            "images": torch.as_tensor(self.images[indices], dtype=torch.uint8),
            "num_clicks_per_object": clip_num_clicks_per_target,
            "fg_coords_list": clip_fg_coords_list,
            "bg_coords_list": clip_bg_coords_list,
            "max_timestamp_list": clip_max_timestamps,
            "indices": _indices,
            "orig_to_serial_id": clip_orig_to_serial_id,
            "serial_to_orig_id": clip_serial_to_orig_id,
            # extras
            "panoptic_masks": self.gt_masks[indices]
        }

        self.prev_clip_input = inputs
        return inputs
    

    def record_click(self, frame_idx, obj_id, coords):
        """
        Record a click in global buffers and update `not_clicked_map` at the clicked location
        Strategy is specified by `self.sampling_strategy`
        """
        if obj_id == self.bg_id:
            # record a bg click
            self.bg_coords_list[frame_idx].append(([coords[0], coords[1], -1, frame_idx, self.t]))
            self.all_bg_clicks[frame_idx].append(([coords[0], coords[1], -1, frame_idx, self.t]))
        else:
            # record a fg click
            self.fg_coords_list[frame_idx].append([coords[0], coords[1], obj_id, frame_idx, self.t])
            self.all_fg_clicks[frame_idx].append([coords[0], coords[1], obj_id, frame_idx, self.t])
            self.num_clicks_per_object[frame_idx][obj_id-1] += 1
        
        self.num_clicks_per_frame[frame_idx] += 1
        self.max_timestamps[frame_idx] = self.t
        self.t+=1

        # update not_clicked_map
        if self.sampling_strategy == 0:
            self.not_clicked_map[frame_idx][coords[0], coords[1]] = False
        else:
            assert self.sampling_strategy == 1
            _pm = create_circular_mask(self.H, self.W, centers=[[coords[0], coords[1]]], radius=self.click_radius)
            self.not_clicked_map[frame_idx][np.where(_pm)] = False
        

    def store_prediction(
            self, 
            binary_pred_masks, 
            overlap,
    ):
        """
        Store predicted masks of a clip in the whole sequence

        Args:
            binary_pred_masks: T,N,H,W predicted binary masks
        """
        indices = self.prev_clip_input["indices"]
        if len(indices) >= 2 and indices[1] < indices[0]:
            indices = self.prev_clip_input["indices"][::-1]
        
        # ignore masks
        clip_ignore_masks = self.ignore_masks[indices]
        
        T,N,H,W = binary_pred_masks.shape

        
        panoptic_pred_masks = np.full((T,H,W), fill_value=self.bg_id).astype(np.uint8)
        # compute the panoptic map for each frame
        for fr_idx in range(T):
            for obj_id in range(N):
                # predicted binary mask of current object
                mask = binary_pred_masks[fr_idx][obj_id]
                if mask.any():
                    # original ID of the object in the whole sequence
                    orig_id = self.prev_clip_input["serial_to_orig_id"][obj_id+1]
                    panoptic_pred_masks[fr_idx][np.where(mask==1)] = orig_id
                
            panoptic_pred_masks[fr_idx][np.where(clip_ignore_masks[fr_idx])] = self.bg_id
                
        # save in buffer
        self.pred_masks[indices] = panoptic_pred_masks

        # compute IoU
        for fr_idx in indices:
            self.frame_level_ious[fr_idx] = self.compute_iou(fr_idx)
        
        if self.vis_path is not None:
            self.save_visualization(indices)

        if self.num_overlapping_frames > 0 and overlap:
            # store information about the predicted objects in the overlapping frames
            frames_to_sample = {}
            for fr_idx, tgt_ids in overlap["frames"].items():
                frames_to_sample[fr_idx] = [self.prev_clip_input["serial_to_orig_id"][i+1] for i in tgt_ids]
            
            self.prev_clip_output = {
                "frames": frames_to_sample,
                "masks": overlap["masks"],
            }
        return panoptic_pred_masks


    def save_visualization(self, indices=None, alpha = 0.5):
        """
        Save predicted mask visualization to the disc

        Args:
            vis_path: str, path to the directory where visualizations are to be saved
            round_num: int, current round number
            indices: (optional) list, save visualizations for specified indices # TODO
        """
        vis_path = os.path.join(self.vis_path, self.sequence.id)
        os.makedirs(vis_path, exist_ok=True)
        
        vis_path = os.path.join(vis_path, str(self.round_num))
        if not os.path.isdir(vis_path):
            os.makedirs(vis_path)

        save_masks = self.pred_masks[indices].copy()
        
        for fr_idx, fr_msk in zip(indices, save_masks):
            iou = self.frame_level_ious[fr_idx]
            
            im = self.images[fr_idx].transpose(1,2,0)
            if self.resize:
                im = cv2.resize(im, (self.orig_W, self.orig_H))
                fr_msk = np.resize(fr_msk, (self.orig_H, self.orig_W))

            fr_msk = Image.fromarray(fr_msk.astype(np.uint8))
            fr_msk.putpalette(color_map)
            fr_msk.save(os.path.join(vis_path, f"mask_{fr_idx}_iou_{iou}.png"))
            
            # Convert both to BGR (for OpenCV)
            image_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            fr_msk = np.array(fr_msk.convert("RGB"))
            mask_bgr = cv2.cvtColor(fr_msk, cv2.COLOR_RGB2BGR)
            overlaid = cv2.addWeighted(image_bgr, 1 - alpha, mask_bgr, alpha, 0)
            
            # display clicks            
            if len(self.all_fg_clicks[fr_idx]) > 0:
                show_points(overlaid, self.all_fg_clicks[fr_idx], 1)
            if len(self.all_bg_clicks[fr_idx]) > 0:
                show_points(overlaid, self.all_bg_clicks[fr_idx], 0)
            
            cv2.imwrite(os.path.join(vis_path, f"overlaid_{fr_idx}_iou_{iou}.png"), overlaid)

    def compute_iou(self, frame_idx):
        """
        Compute IoU score of specified frame

        If an object is not present in either gt or pred, it does not contribute to the calculation
        """
        pred = self.pred_masks[frame_idx]
        gt = self.gt_masks[frame_idx]

        # objects in the sequence, not including VOID
        for obj_id in self.object_ids:
            
            g = (gt == obj_id).astype('uint8')
            p = (pred == obj_id).astype('uint8')
            intersection = np.logical_and(p, g).sum()
            union = np.logical_or(p,g).sum()
            if union > 0:
                iou = intersection/union
            else:
                iou = 1.
            
            self.ious[frame_idx][obj_id-1] = iou

        frame_iou = self.ious[frame_idx]
        return frame_iou[np.where(frame_iou>=0)].mean()


    def get_gt_clicks(self):
        """
        For each target, sample a click at its center, in the frame where it first appears.
        NOTE: self.object_appearance contains entries for frames where at least one target
        has made its first appearance
        """
        for fr_idx, new_object_ids in self.object_appearance.items():
            # panoptic mask of the frame
            fr_mask = self.gt_masks[fr_idx]

            for obj_id in new_object_ids:
                # binary object mask
                obj_mask = (fr_mask==obj_id).astype(np.uint8)
                # sample a click at the object center
                center_coords = get_center_coords(obj_mask * self.not_clicked_map[fr_idx])
                # record the click
                self.record_click(fr_idx, obj_id, center_coords)
    
    
    def get_corrective_click(self, frame_idx, obj_id, padding=True):
        """
        Obtain a corrective click on the specified object in the specified frame

        Args:
            frame_idx: int, frame to sample the click on
            obj_id: int, ID of the object to sample the click on
            padding: bool (default: True), whether to apply padding before `cv2.distanceTransform`
        """
        gt_instance_mask = np.asarray(self.gt_masks[frame_idx] == obj_id, dtype=np.bool_)
        pred_instance_mask = np.asarray(self.pred_masks[frame_idx] == obj_id, dtype=np.bool_)

        H,W = gt_instance_mask.shape
        timestamp = max(self.max_timestamps)

        # negative error map - g.t. foreground missed by the prediction
        fn_mask = np.logical_and(gt_instance_mask, np.logical_not(pred_instance_mask))
        # positive error map - g.t. background covered by the prediction
        fp_mask = np.logical_and(np.logical_not(gt_instance_mask), pred_instance_mask)

        # distance transform to find the center of the error region
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
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

        sample_locations = [[coords_y[0], coords_x[0]]]
        obj_index = self.gt_masks[frame_idx][coords_y[0], coords_x[0]]

        if self.sampling_strategy == 0:
            self.not_clicked_map[frame_idx][[coords_y[0], coords_x[0]]] = False
        elif self.sampling_strategy == 1:
            _pm = create_circular_mask(H,W, centers=sample_locations, radius=self.click_radius)
            self.not_clicked_map[frame_idx][np.where(_pm==1)] = False
        else:
            raise NotImplementedError
        
        if obj_index == self.bg_id:
            self.bg_coords_list[frame_idx].append([coords_y[0], coords_x[0], obj_index, frame_idx, timestamp+1])
        else:
            obj_index = self.orig_to_serial_ids[obj_index] - 1
            self.fg_coords_list[frame_idx][obj_index].append([coords_y[0], coords_x[0], obj_index, frame_idx, timestamp+1])
            self.num_clicks_per_object[frame_idx][obj_index] += 1
        
        self.max_timestamps[frame_idx] = timestamp+1
        self.num_clicks_per_frame[frame_idx] += 1
        return obj_index
    

    def find_refinement_target(self, iou_threshold):
        """
        Find the object with the worst IoU over the sequence
        Find the frame where this object has the worst IoU
        """
        avg_obj_ious = []
        for orig_id in self.object_ids:
            serial_id = self.orig_to_serial_ids[orig_id]
            
            obj_ious = self.ious[:, serial_id-1]         # T
            avg_obj_ious.append(obj_ious[np.where(obj_ious>=0)].mean())

        min_iou = min(avg_obj_ious)
        if min_iou >= iou_threshold:
            return -1, -1
        
        min_obj_idx = np.asarray(avg_obj_ious).argmin()
        ious_obj_id_to_refine = self.ious[:, min_obj_idx]
        frames_present = np.where(ious_obj_id_to_refine>=0)
        frame_idx = ious_obj_id_to_refine[frames_present].argmin()
        lowest_frame_idx = frames_present[0][frame_idx]

        orig_obj_id_to_refine = self.serial_to_orig_ids[min_obj_idx+1]

        return lowest_frame_idx, orig_obj_id_to_refine
    

    def get_binary_gt_masks(self):
        ...

        
