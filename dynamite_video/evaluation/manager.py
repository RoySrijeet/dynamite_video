import cv2
import numpy as np
import os
import torch

from PIL import Image

from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks, serialize_object_ids, convert_binary_to_panoptic, convert_panoptic_to_binary
from dynamite_video.evaluation.eval_utils import create_circular_mask, color_map, show_points, get_center_coords

class SequenceManager:
    """
    Sequence manager
    """

    def __init__(self, sequence, dataset_meta, tfms):
        """
        Initialize with information on the sequence data, including ground truth masks

        Sequence Manager maintains all object-related info in their original ID, which
        follows the format: semantic_label * max_instances_per_category + instance_id

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
        # ignore class label in panoptic masks
        self.ignore_label = self.ignore_class * self.max_instances_per_category
        
        # sequence level information
        
        # `GenericVideoSequence` instance of current video
        self.sequence = sequence
        # num of frames in the sequence
        self.T = len(self.sequence)
        # num of targets in the sequence, ignore VOID mask
        self.N = len(self.sequence.object_ids) - 1
        # original spatial resolution
        self.orig_H, self.orig_W = self.sequence.image_dims
        
        # load tensors
        
        # load images - T,3,orig_H,orig_W
        self.images = self.sequence.load_images()
        # load ground truth panoptic masks - T,orig_H,orig_W
        self.gt_masks = self.sequence.prepare_eval_masks(fill_value=self.ignore_label)
        # apply transformation on image and gt mask tensors
        self.H, self.W = self.compute_tfm_sizes(tfms)
        # ignore mask - T,H,W
        self.ignore_masks = (self.gt_masks==self.ignore_label)

        # object level information

        # IDs of the objec
        self.object_ids = self.sequence.object_ids
        if self.ignore_label in self.object_ids:
            self.object_ids.remove(self.ignore_label)
        # serialize object IDs
        self.orig_to_serial_ids, self.serial_to_orig_ids = serialize_object_ids(self.object_ids)
        # ensure there's no intersection between original IDs and the serialized IDs
        assert set(self.orig_to_serial_ids.keys()).intersection(set(self.orig_to_serial_ids.values())) == set()
        # maintain a record to keep track of objects that were already discovered
        self.object_discovery = set()

        # clip level information

        # length of clips to be extracted from the sequence
        self.clip_length = dataset_meta["clip_length"]
        # overlap between successive clips
        self.num_overlapping_frames = dataset_meta["num_overlapping_frames"]

        # click level information
        
        # click radius - region around a existing click that is excluded when sampling a new click
        self.click_radius = 5
        # Strategy to avoid regions while sampling next clicks
        # 0: new click avoids all the previously sampled click locations
        # 1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks
        self.sampling_strategy = 1
        # allow sampling a click if object area is larger than a threshold
        self.MIN_MASK_AREA = 0
        # maintain a map of regions to avoid during click sampling, initialized with ignore mask regions
        self.not_clicked_map = (self.gt_masks!=self.ignore_label).astype(np.bool_)
        # foreground clicks sampled on each object in each frame
        self.fg_coords_list = [[[] for _ in range(self.N)] for _ in range(self.T)]
        # background clicks sampled on each frame
        self.bg_coords_list = [[] for _ in range(self.T)]
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
        self.pred_masks = np.zeros((self.T, self.H, self.W), dtype=np.uint32)
        # store IoU
        self.ious = np.full((self.T, self.N), fill_value=-1., dtype=np.single)
        self.frame_level_ious = np.zeros((self.T), dtype=np.single)

        
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
            self.gt_masks = resize_masks(self.gt_masks, new_width, new_width, binary=False)
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
        Extract a clip from the sequence specified by the indices. The clip could be reversed
        in time.

        Args:
            _indices: list(int), list of frame indices

        Returns:
            input: dict, compatible with `inputs` argument in `DynamiteModel.forward()`
        """
        
        indices = _indices
        if len(indices) >= 2 and indices[1] < indices[0]:
            indices = _indices[::-1]
        
        # panoptic maps of the clip frames, including `ignore_label` - T,H,W
        clip_gt_masks = self.gt_masks[indices[0]:indices[-1]+1]

        # objects present in the ground truth panoptic mask
        clip_orig_ids = list(np.unique(clip_gt_masks))
        # remove `ignore_label`, if present
        if self.ignore_label in clip_orig_ids:
            clip_orig_ids.remove(self.ignore_label)
        
        # serialize object IDs
        clip_orig_to_serial_id, clip_serial_to_orig_id = serialize_object_ids(clip_orig_ids)
        assert set(clip_orig_to_serial_id.keys()).intersection(set(clip_orig_to_serial_id.values())) == set()

        # panoptic to serialized binary gt masks, ** excluding ** VOID
        clip_binary_gt_masks, clip_objects_per_frame = convert_panoptic_to_binary(clip_gt_masks, clip_orig_to_serial_id, ignore=self.ignore_label)

        # for each object, find the frame where it is the largest
        frame_to_sample_from = clip_binary_gt_masks.sum((2,3)).argmax(0)
        
        # # for each object, find the frame where it first appeared
        # frame_to_sample_from = clip_binary_gt_masks.any((2,3)).argmax(0)
        
        # record clicks for the frames
        clip_T = len(indices)
        clip_N = len(clip_orig_ids)
        clip_fg_coords_list = [[[] for _ in range(clip_N)] for _ in range(len(indices))]
        clip_num_clicks_per_object = np.zeros((clip_T, clip_N), dtype=np.uint16)

        for global_obj_id, local_fr_idx in zip(clip_orig_ids, frame_to_sample_from):

            if global_obj_id in self.object_discovery:
                # do not sample a click if the object has been already discovered in a previous frame
                continue
            
            local_obj_id = clip_orig_to_serial_id[global_obj_id]
            global_fr_idx = indices[local_fr_idx]
            
            # ground truth binary mask of the object in the frame
            obj_mask = clip_binary_gt_masks[local_fr_idx][local_obj_id-1]

            sm = obj_mask.sum()
            if obj_mask.sum() < self.MIN_MASK_AREA:
                continue
            
            center_coords = get_center_coords(obj_mask * self.not_clicked_map[global_fr_idx])
            
            # serialized object ID in the clip
            clip_fg_coords_list[local_fr_idx][local_obj_id-1].append([center_coords[0], center_coords[1], local_obj_id, local_fr_idx, self.t])
            clip_num_clicks_per_object[local_fr_idx][local_obj_id-1] += 1

            self.record_click(global_fr_idx, global_obj_id, center_coords)
            self.object_discovery.add(global_obj_id)

        # overlapping frames
        if self.num_overlapping_frames > 0:
            overlapping_frame_indices = sorted(_indices[:self.num_overlapping_frames])
            overlapping_frame_preds = np.stack(self.pred_masks[overlapping_frame_indices])

            if overlapping_frame_preds.any():
                overlapping_objects_predicted = list(np.unique(overlapping_frame_preds))

                # for each object, randomly pick one frame to sample a click from
                for global_obj_id in overlapping_objects_predicted:
                    if global_obj_id == self.ignore_label:
                        continue

                    # find the frames where the object was predicted in and randomly select one
                    fr_idx = (overlapping_frame_preds==global_obj_id).astype(np.uint8).sum(axis=(1,2)) > 0
                    fr_idx = np.random.choice(np.where(fr_idx)[0])

                    global_fr_idx = overlapping_frame_indices[fr_idx]
                    local_fr_idx = indices.index(global_fr_idx)

                    obj_mask = (overlapping_frame_preds[fr_idx]==global_obj_id).astype('uint8')
                    center_coords = get_center_coords(obj_mask)

                    # serialized object ID in the clip
                    local_obj_id = clip_orig_to_serial_id[global_obj_id]
                    clip_fg_coords_list[local_fr_idx][local_obj_id-1].append([center_coords[0], center_coords[1], local_obj_id, local_fr_idx, self.t])
                    clip_num_clicks_per_object[local_fr_idx][local_obj_id-1] += 1

                    self.record_click(global_fr_idx, global_obj_id, center_coords)
        
        # input to model forward pass
        input = {
            "images": torch.as_tensor(self.images[indices[0]:indices[-1]+1], dtype=torch.uint8),
            "objects_per_frame": clip_objects_per_frame,
            "num_clicks_per_object": clip_num_clicks_per_object,
            "fg_coords_list": clip_fg_coords_list,
            "bg_coords_list": [[] for _ in range(len(indices))],
            "max_timestamp_list": self.max_timestamps[indices[0]:indices[-1]+1],
        }

        clip = {
            "indices": _indices,
            "orig_to_serial_id": clip_orig_to_serial_id,
            "serial_to_orig_id": clip_serial_to_orig_id,
            "panoptic_gt_masks": clip_gt_masks,
            "binary_gt_masks": clip_binary_gt_masks,
        }
        
        return clip, input
    

    def record_click(self, frame_idx, obj_id, coords):
        """
        Record a click in global buffers and update `not_clicked_map` at the clicked location
        Strategy is specified by `self.sampling_strategy`
        """
        # obj_id still has the format: semantic_map * max_instances_per_category + instance_map
        obj_id = self.orig_to_serial_ids[obj_id]
        if self.sampling_strategy == 0:
            self.not_clicked_map[frame_idx][coords[0], coords[1]] = False
        elif self.sampling_strategy == 1:
            _pm = create_circular_mask(self.H, self.W, centers=[[coords[0], coords[1]]], radius=self.click_radius)
            self.not_clicked_map[frame_idx][np.where(_pm)] = False
        else:
            raise NotImplementedError

        self.fg_coords_list[frame_idx][obj_id-1].append([coords[0], coords[1], obj_id, frame_idx, self.t])
        self.num_clicks_per_object[frame_idx][obj_id-1] += 1
        self.num_clicks_per_frame[frame_idx] += 1
        self.max_timestamps[frame_idx] = self.t
        self.t+=1
        

    def store_prediction(self, binary_pred_masks, clip):
        """
        Store predicted masks of a clip in the whole sequence

        Args:
            binary_pred_masks: T,N,H,W predicted binary masks
            clip: GenericVideoSequence
            indices: indices w.r.t whole sequence
        """
        indices = clip["indices"]
        panoptic_pred_masks = convert_binary_to_panoptic(binary_pred_masks, clip["serial_to_orig_id"], fill_value=self.ignore_label)
        
        # ignore masks
        clip_ignore_masks = self.ignore_masks[indices]
        panoptic_pred_masks[np.where(clip_ignore_masks)] = self.ignore_label

        self.pred_masks[indices] = panoptic_pred_masks

        # compute IoU
        for fr_idx in indices:
            self.frame_level_ious[fr_idx] = self.compute_iou(fr_idx)

        return panoptic_pred_masks


    def save_visualization(self, vis_path, round_num, indices=None, alpha = 0.5):
        """
        Save predicted mask visualization to the disc

        Args:
            vis_path: str, path to the directory where visualizations are to be saved
            round_num: int, current round number
            indices: (optional) list, save visualizations for specified indices # TODO
        """
        vis_path = os.path.join(vis_path, self.sequence.id)
        os.makedirs(vis_path, exist_ok=True)
        
        vis_path = os.path.join(vis_path, str(round_num))
        if not os.path.isdir(vis_path):
            os.makedirs(vis_path)

        save_masks = np.zeros_like(self.pred_masks[indices], dtype=np.uint8)
        
        # pred_mask has labels in the format: semantic_map * max_instances_per_category + instance_map
        # for easier visualization, the object IDs are serialized
        obj_ids = np.unique(self.pred_masks[indices]).tolist()
        if self.ignore_label in obj_ids:
            obj_ids.remove(self.ignore_label)
        for i in obj_ids:
            save_masks[np.where(save_masks==i)] = self.orig_to_serial_ids[i]
        
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
            # image_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            # fr_msk = np.array(fr_msk.convert("RGB"))
            # mask_bgr = cv2.cvtColor(fr_msk, cv2.COLOR_RGB2BGR)
            # overlayed = cv2.addWeighted(image_bgr, 1 - alpha, mask_bgr, alpha, 0)
            
            # # display clicks
            # fg_coords = self.fg_coords_list[fr_idx]
            # flattened_fg_coords = []
            # for coords in fg_coords:
            #     flattened_fg_coords.extend(coords)
            # if len(flattened_fg_coords) > 0:
            #     show_points(overlayed, flattened_fg_coords, 1)
            # if len(self.bg_coords_list[fr_idx]) > 0:
            #     show_points(overlayed, self.bg_coords_list[fr_idx], 0)
            
            # cv2.imwrite(os.path.join(vis_path, f"overlayed_{fr_idx}_iou_{iou}.png"), overlayed)

    def compute_iou(self, frame_idx):
        """
        Compute IoU score of specified frame
        """
        pred = self.pred_masks[frame_idx]
        gt = self.gt_masks[frame_idx]

        # objects in the sequence, not including VOID
        objects = self.object_ids
        
        for orig_id in objects:
            
            g = (gt == orig_id).astype('uint8')
            p = (pred == orig_id).astype('uint8')
            intersection = np.logical_and(p, g).sum()
            union = np.logical_or(p,g).sum()
            if union > 0:
                iou = intersection/union

                serial_id = self.orig_to_serial_ids[orig_id]
                self.ious[frame_idx][serial_id-1] = iou

        frame_iou = self.ious[frame_idx]
        return frame_iou[np.where(frame_iou>=0)].mean()


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
        
        if obj_index == -1:
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

        
