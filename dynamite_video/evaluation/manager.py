import cv2
import numpy as np
import os
import torch

from PIL import Image

from dynamite_video.data.utils.clicker import get_center_coords
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks, serialize_object_ids
from dynamite_video.evaluation.eval_utils import create_circular_mask, color_map, show_points

class SequenceManager:
    """
    Sequence manager
    """

    def __init__(self, sequence, dataset_meta, tfms):
        """
        Initialize with information on the sequence data, including ground truth masks

        Args:
            sequence: `GenericVideoSequence` instance of current video
            dataset_meta: dict with dataset level info on following properties:
                * category_labels: mapping between category ID and label, e.g., 0: 'road'
                * clip_length: int, length of each sub-sequence
                * num_overlapping_frames: int, overlap between consecutive sub-sequences
                * fps: sequence FPS
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
        # length of clips to be extracted from the sequence
        self.clip_length = dataset_meta["clip_length"]
        # overlap between successive clips
        self.num_overlapping_frames = dataset_meta["num_overlapping_frames"]
        # sequence level information
        self.sequence = sequence
        # dimensions T,N,H,W
        self.T = len(self.sequence)
        self.N = len(self.sequence.object_ids) - 1 # ignoring ignore mask
        self.orig_H, self.orig_W = self.sequence.image_dims
        
        # load images and ground truth masks
        self.images = self.sequence.load_images()
        # gt semantic masks follow the labelling format: semantic_map * max_instances_per_category + instance_map
        self.gt_masks = self.sequence.prepare_eval_masks()
        # transformations
        self.H, self.W = self.compute_tfm_sizes(tfms)
        self.ignore_masks = (self.gt_masks==self.ignore_class * self.max_instances_per_category).astype(np.uint8)

        # maintain serialized object IDs
        self.orig_to_serial_ids, self.serial_to_orig_ids = serialize_object_ids(self.sequence.object_ids)
        # ensure there's no intersection between original IDs and the serialized IDs
        assert set(self.orig_to_serial_ids.keys()).intersection(set(self.orig_to_serial_ids.values())) == set()
        # whether an object was already discovered or not
        self.object_discovery = set()

        # click radius - region around a existing click that is excluded when sampling a new click
        self.click_radius = 5
        # Strategy to avoid regions while sampling next clicks
        self.sampling_strategy = 1
        # maintain a map of regions to avoid during click sampling, initialized with ignore mask regions
        self.not_clicked_map = (self.gt_masks!=self.ignore_class * self.max_instances_per_category).astype(np.bool_)

        # initialize buffers
        # foreground clicks sampled on each object in each frame
        self.fg_coords_list = [[[] for _ in range(self.N)] for _ in range(self.T)]
        # background clicks sampled on each frame
        self.bg_coords_list = [[] for _ in range(self.T)]
        # time stamp of the latest click on each frame
        self.max_timestamps = np.zeros((self.T), dtype=np.uint16)
        # num clicks on each object in each frame
        self.num_clicks_per_object = np.zeros((self.T, self.N), dtype=np.uint16)
        # first click at timestamp 1
        self.t = 1
        
        # to store predicted masks
        self.pred_masks = np.zeros((self.T, self.H, self.W), dtype=np.uint32)
        # store frame-level IoU
        self.ious = np.zeros(self.T)

        self.MIN_MASK_AREA = 400
        
    
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
        Given a list of frame indices, extract a clip consisting of these indices 
        from the whole sequence.

        Args:
            _indices: list(int)

        Returns:
            clip: dict, compatible with `inputs` argument in `DynamiteModel.forward()`
        """
        indices = _indices
        if len(indices) >= 2 and indices[1] < indices[0]:
            indices = _indices[::-1]

        # semantic maps of the clip frames - T,H,W
        clip_gt_masks = self.gt_masks[indices[0]:indices[-1]+1]

        # serialize object IDs in the clip
        clip_orig_ids = list(np.unique(clip_gt_masks))
        clip_orig_to_serial_id, clip_serial_to_orig_id = serialize_object_ids(clip_orig_ids)
        assert set(clip_orig_to_serial_id.keys()).intersection(set(clip_orig_to_serial_id.values())) == set()
        
        # sample gt clicks - only if there is any object appearing in the clip for the first time
        clip_fg_coords_list = [[[] for _ in range(len(clip_orig_ids))] for _ in range(len(indices))]
        clip_num_clicks_per_object = np.zeros((len(indices), len(clip_orig_ids)), dtype=np.uint16)
        clip_objects_per_frame = []
        
        for local_fr_idx, global_fr_idx in enumerate(indices):
            
            # gt mask of current frame - H,W
            fr_mask = clip_gt_masks[local_fr_idx].copy()
            fr_obj_ids = list(np.unique(fr_mask))
            
            # serialize the object IDs in this frame
            clip_objects_per_frame.append([clip_orig_to_serial_id[obj_id] for obj_id in fr_obj_ids])

            for global_obj_id in fr_obj_ids:
                if global_obj_id in self.object_discovery:
                    continue
                
                if global_obj_id == self.ignore_class * self.max_instances_per_category:
                    # do not sample click for ignore class
                    continue

                # ground truth binary mask of the object in the frame
                obj_mask = (fr_mask == global_obj_id).astype(np.uint8)

                self.object_discovery.add(global_obj_id)
                
                center_coords = get_center_coords(obj_mask)
                # serialized object ID in the clip
                local_obj_id = clip_orig_to_serial_id[global_obj_id]
                clip_fg_coords_list[local_fr_idx][local_obj_id-1].append([center_coords[0], center_coords[1], local_obj_id, local_fr_idx, self.t])
                clip_num_clicks_per_object[local_fr_idx][local_obj_id-1] += 1

                self.record_click(global_fr_idx, global_obj_id, center_coords)

        # sample from overlapping frames in the clip
        overlapping_frame_indices = sorted(_indices[:self.num_overlapping_frames])
        overlapping_frame_preds = np.stack(self.pred_masks[overlapping_frame_indices])
        
        if overlapping_frame_preds.any():
            overlapping_objects_predicted = list(np.unique(overlapping_frame_preds))

            # for each object, randomly pick one frame to sample a click from
            for global_obj_id in overlapping_objects_predicted:
                if global_obj_id == self.ignore_class * self.max_instances_per_category:
                    continue

                fr_idx = (overlapping_frame_preds==global_obj_id).astype(np.uint8).sum(axis=(1,2)) > 0
                fr_idx = np.random.choice(np.where(fr_idx)[0])
                global_fr_idx = overlapping_frame_indices[fr_idx]
                local_fr_idx = indices.index(global_fr_idx)

                obj_mask = (overlapping_frame_preds[fr_idx]==global_obj_id).astype('uint8')
                # obj_mask = (clip_gt_masks[fr_idx]==global_obj_id).astype('uint8')
                center_coords = get_center_coords(obj_mask)

                # serialized object ID in the clip
                local_obj_id = clip_orig_to_serial_id[global_obj_id]
                clip_fg_coords_list[local_fr_idx][local_obj_id-1].append([center_coords[0], center_coords[1], local_obj_id, local_fr_idx, self.t])
                clip_num_clicks_per_object[local_fr_idx][local_obj_id-1] += 1

                self.record_click(global_fr_idx, global_obj_id, center_coords)
            
        clip = {
            "indices": _indices,
            "orig_to_serial_id": clip_orig_to_serial_id,
            "serial_to_orig_id": clip_serial_to_orig_id,
        }
        # input to model forward pass
        input = {
            "images": torch.as_tensor(self.images[indices[0]:indices[-1]+1], dtype=torch.uint8),
            "objects_per_frame": clip_objects_per_frame,
            "num_clicks_per_object": clip_num_clicks_per_object,
            "fg_coords_list": clip_fg_coords_list,
            "bg_coords_list": [[] for _ in range(len(indices))],
            "max_timestamp_list": self.max_timestamps[indices[0]:indices[-1]+1],
        }
        
        return clip, input
    

    def record_click(self, frame_idx, obj_id, coords):
        """
        Record a click in global buffers and update `not_clicked_map` at the clicked location
        Strategy is specified by `self.sampling_strategy`
            0: new click avoids all the previously sampled click locations
            1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks
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
        self.max_timestamps[frame_idx] = self.t
        self.t+=1
        

    def store_prediction(self, clip_preds, clip, indices):
        """
        Store predicted masks of a clip in the whole sequence

        Args:
            clip_preds: list of N,H,W predicted binary masks
            clip: GenericVideoSequence
            indices: indices w.r.t whole sequence
        """
        # convert binary masks to panoptic
        pred_sem_masks = np.zeros((len(indices), self.H, self.W), dtype=np.uint32) # T,H,W

        for local_fr_idx, fr_pred in enumerate(clip_preds):
            for i, msk in enumerate(fr_pred):
                pred_sem_masks[local_fr_idx][np.where(msk == 1)] = clip["serial_to_orig_id"][i+1]

        # ignore masks
        clip_ignore_masks = self.ignore_masks[indices]
        pred_sem_masks[np.where(clip_ignore_masks==1)] = self.ignore_class * self.max_instances_per_category

        self.pred_masks[indices] = pred_sem_masks
        return pred_sem_masks


    def save_visualization(self, vis_path, round_num=0, indices=None, alpha = 0.5):
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

        save_masks = self.pred_masks[indices].copy()
        
        # pred_mask has labels in the format: semantic_map * max_instances_per_category + instance_map
        # for easier visualization, the object IDs are serialized
        obj_ids = np.unique(save_masks)
        for i in obj_ids:
            save_masks[np.where(save_masks==i)] = self.orig_to_serial_ids[i]
        
        for fr_idx, fr_msk in zip(indices, save_masks):
            iou = self.compute_iou(fr_idx)
            
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
            overlayed = cv2.addWeighted(image_bgr, 1 - alpha, mask_bgr, alpha, 0)
            
            # display clicks
            fg_coords = self.fg_coords_list[fr_idx]
            flattened_fg_coords = []
            for coords in fg_coords:
                flattened_fg_coords.extend(coords)
            if len(flattened_fg_coords) > 0:
                show_points(overlayed, flattened_fg_coords, 1)
            if len(self.bg_coords_list[fr_idx]) > 0:
                show_points(overlayed, self.bg_coords_list[fr_idx], 0)
            
            cv2.imwrite(os.path.join(vis_path, f"overlayed_{fr_idx}_iou_{iou}.png"), overlayed)


    def compute_iou(self, frame_idx, ignore_small=True):
        """
        Compute IoU score of specified frame
        """
        pred = self.pred_masks[frame_idx]
        gt = self.gt_masks[frame_idx]

        objects = np.unique(gt)
        ious = []

        for obj_id in objects:
            if obj_id == self.ignore_class * self.max_instances_per_category:
                continue
            
            g = (gt == obj_id).astype('uint8')
            if ignore_small and g.sum() < 200:
                continue
            
            p = (pred == obj_id).astype('uint8')
            intersection = np.logical_and(p, g).sum()
            union = np.logical_or(p,g).sum()

            if union == 0:
                ious.append(1.)
                continue
            
            ious.append(intersection/union)
        
        self.ious[frame_idx] = round(sum(ious)/len(ious),5)
        return self.ious[frame_idx]


    def get_corrective_click(self, frame_idx, obj_id, padding=True):
        """
        Obtain a corrective click on the specified object in the specified frame

        Args:
            frame_idx: int, frame to sample the click on
            obj_id: int, ID of the object to sample the click on
            padding: bool (default: True), whether to apply padding before `cv2.distanceTransform`
        """
        gt_instance_mask = np.asarray(self.gt_binary_masks[frame_idx][obj_id], dtype=np.bool_)
        pred_instance_mask = np.asarray(self.pred_masks[frame_idx][obj_id], dtype=np.bool_)

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
        obj_index = self.gt_semantic_masks[frame_idx][coords_y[0], coords_x[0]] - 1

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
            self.fg_coords_list[frame_idx][obj_index].append([coords_y[0], coords_x[0], obj_index, frame_idx, timestamp+1])
        
        self.max_timestamps[frame_idx] = timestamp+1
        self.num_clicks_per_object[frame_idx][obj_index] += 1
        self.num_clicks_per_frame[frame_idx] += 1
        return obj_index