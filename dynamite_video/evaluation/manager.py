import os
import cv2
import torch
import random
import numpy as np

from PIL import Image

from dynamite_video.data.utils.clicker import get_center_coords
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.evaluation.eval_utils import davis_palette, create_circular_mask

class SequenceManager:
    """
    Sequence manager
    """

    def __init__(self, sequence, dataset_meta, tfms):
        """
        Initialize with information on the sequence data, including ground truth masks

        Args:
            sequence: GenericVideoSequence with following properties:
                * id: sequence ID
                * height: original height of sequence frames
                * width: original width of sequence frames
                * image_dims: (height, width)
                * path_to_images: absolute path to the directory containing RGB images
                * image_paths: relative path to individual frame RGBs
                * max_class_id: maximum ID of the semantic classes in the sequence
                * object_ids: IDs of objects present in the sequence
                * object_categories: mapping between object ID and category ID
                * orig_to_serial_id: mapping between original and serialized object IDs
                * serial_to_orig_id: mapping between serialized and original object IDs
                * segmentations: list of dicts, where each dict is an (object ID, RLE) pair

            dataset_meta: dict with dataset level info on following properties:
                * category_labels: mapping between category ID and label, e.g., 0: 'road'
                * clip_length: int, length of each sub-sequence
                * num_overlapping_frames: int, overlap between consecutive sub-sequences
                * fps: sequence FPS
            
            tfms: transformation info from config, cfg.INPUT
        """
        self.sequence = sequence
        
        self.clip_length = dataset_meta["clip_length"]
        self.num_overlapping_frames = dataset_meta["num_overlapping_frames"]
        self.category_labels = dataset_meta["category_labels"]
        self.fps = dataset_meta["fps"]    

        self.object_discovery = self.sequence.get_object_discovery()

        # resolutions
        self.T = len(self.sequence)
        self.N = len(self.sequence.object_ids)
        self.orig_H = self.sequence.height
        self.orig_W = self.sequence.width
        self.H, self.W, self.resize, self.rgb = self.compute_tfm_sizes(tfms)

        # click radius - region around a existing click that is excluded when sampling a new click
        self.click_radius = 5
        # Strategy to avoid regions while sampling next clicks
        # 0: new click avoids all the previously sampled click locations
        # 1: new click avoids all locations upto radius=self.click_radius around all prev sampled clicks
        self.sampling_strategy = 1
        self.not_clicked_map = np.ones((self.T, self.H, self.W), dtype=np.bool_)
        self.t = 1

        # initialize buffers
        # foreground clicks sampled on each object in each frame
        self.fg_coords_list = [[[] for _ in range(self.N)] for _ in range(self.T)]
        # background clicks sampled on each frame
        self.bg_coords_list = [[] for _ in range(self.T)]
        # time stamp of the latest click on each frame
        self.max_timestamps = np.zeros(self.T).astype(np.uint16)
        # num clicks on each object in each frame
        self.num_clicks_per_object = np.zeros((self.T, self.N), dtype=np.uint16)
        # num clicks per frame
        self.num_clicks_per_frame = np.zeros(self.T).astype(np.uint16)
        
    
    def compute_tfm_sizes(self, tfms):
        """
        Compute transformed resolution

        tfms: transformation info from config, cfg.INPUT
        """
        if tfms.AUGMENTATION.RESIZE_TEST:
            new_height, new_width = compute_resized_dims(self.orig_H, self.orig_W,
                                                        min_dim=tfms.AUGMENTATION.MIN_DIM_TEST,
                                                        max_dim=tfms.AUGMENTATION.MAX_DIM_TEST,
                                                    )
            return new_height, new_width, True, tfms.RGB
        return self.orig_H, self.orig_W, False, tfms.RGB
        

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

        Indices may be incremental or decremental. In case of the latter, the clip
        is temporally reversed. The reversed indices are important in finding the
        overlapping frames, but for the rest of the workflow the indices are reversed
        once more (turning them incremental)

        Args:
            _indices: list(int)

        Returns:
            clip: dict. The format should be consistent with that of the batch input for 
                training pass, i.e., inputs argument in `DynamiteModel.forward()`. See 
                `TraininMapper` for more details.
        """
        indices = _indices
        # indices - always incremental, _indices - true order
        if len(indices) >= 2 and indices[1] < indices[0]:
            indices = _indices[::-1]

        clip = self.sequence.extract_subsequence(indices)

        # load images [T,H,W,3]
        clip_images = clip.load_images()

        # load binary instance masks [T,N,H,W]
        clip_binary_masks, clip_objects_per_frame, clip_object_ids, _ = clip.prepare_masks()

        # transformations
        if self.resize:
            clip_images = resize_images(clip_images, self.H, self.W)
            clip_binary_masks = resize_masks(clip_binary_masks, self.H, self.W, binary=True)

        clip_images = np.transpose(clip_images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
        if self.rgb:                                  # BGR -> RGB
            clip_images = np.flip(clip_images, 1).copy()
        
        # sample clicks
        clip_fg_coords_list = [[[] for _ in range(len(clip.object_ids))] for _ in range(len(clip))]
        clip_bg_coords_list = [[] for _ in range(len(clip))]
        clip_num_clicks_per_object = np.zeros((len(clip), len(clip.object_ids)), dtype=np.uint16)
        clip_max_timestamp_list = np.zeros(len(clip)).astype(np.uint16)
        
        for local_obj_id in clip_object_ids:
            
            # global frame index
            global_obj_id = clip.serial_to_orig_id[local_obj_id]
            global_fr_idx = self.object_discovery[global_obj_id]
            # local frame index
            local_fr_idx = indices.index(global_fr_idx)
            
            # mask of specified object in the frame
            obj_mask = clip_binary_masks[local_fr_idx][local_obj_id]
            
            # center coordinates of the foreground mask
            center_coords = get_center_coords(obj_mask)

            # update not_clicked_map
            if self.sampling_strategy == 0:
                self.not_clicked_map[global_fr_idx][center_coords[0], center_coords[1]] = False
            elif self.sampling_strategy == 1:
                _pm = create_circular_mask(self.H, self.W, centers=[[center_coords[0], center_coords[1]]], radius=self.click_radius)
                self.not_clicked_map[global_fr_idx][np.where(_pm)] = False
            else:
                raise NotImplementedError
            
            clip_fg_coords_list[local_fr_idx][local_obj_id].append([center_coords[0], center_coords[1], local_obj_id, local_fr_idx, self.t])
            clip_num_clicks_per_object[local_fr_idx][local_obj_id-1] += 1
            clip_max_timestamp_list[local_fr_idx] = self.t
            
            # record sampled click
            self.fg_coords_list[global_fr_idx][global_obj_id-1].append([center_coords[0], center_coords[1], global_obj_id, global_fr_idx, self.t])
            self.num_clicks_per_object[global_fr_idx][global_obj_id-1] += 1
            self.num_clicks_per_frame[global_fr_idx] += 1
            self.max_timestamps[global_fr_idx] = self.t
            self.t+=1


        return {
            "images": clip_images,
            "binary_masks": None,
            "semantic_masks": None,
            "padding_mask": None,
            "bg_masks": None,
            "ignore_masks": None,
            # "object_ids": None,
            "objects_per_frame": clip_objects_per_frame,
            # "frame_object_occupancy": None,
            "num_clicks_per_object": clip_num_clicks_per_object,
            "fg_coords_list": clip_fg_coords_list,
            "bg_coords_list": clip_bg_coords_list,
            "max_timestamp_list": clip_max_timestamp_list,
            # "meta": None,
        }
    
    
    def get_gt_clicks(self):
        """
        For each object present in the sequence, go to the frame where the object 
        first appeared and sample a foreground click at its center.

        Populates click buffers with sampled click information
        """
        t = 1
        
        for obj_id, fr_idx in self.object_discovery.items():
            # object with ID `obj_id` first appeared in frame at index `fr_idx``

            obj_mask = self.gt_binary_masks[fr_idx][obj_id-1]

            # center coordinates of the foreground mask
            center_coords = get_center_coords(obj_mask)

            # update not_clicked_map
            if self.sampling_strategy == 0:
                self.not_clicked_map[fr_idx][center_coords[0], center_coords[1]] = False
            elif self.sampling_strategy == 1:
                _pm = create_circular_mask(self.H, self.W, centers=[[center_coords[0], center_coords[1]]], radius=self.click_radius)
                self.not_clicked_map[fr_idx][np.where(_pm)] = False
            else:
                raise NotImplementedError

            # record sampled click
            self.fg_coords_list[fr_idx][obj_id-1].append([center_coords[0], center_coords[1], obj_id, fr_idx, t])
            self.num_clicks_per_object[fr_idx][obj_id-1] += 1
            self.num_clicks_per_frame[fr_idx] += 1
            self.max_timestamps[fr_idx] = t
            t+=1


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


    
    
    
    def extract_clip_2(self, _indices):
        """
        Given a list of frame indices, extract a clip consisting of these indices 
        from the whole sequence.

        Indices may be incremental or decremental. In case of the latter, the clip
        is temporally reversed. The reversed indices are important in finding the
        overlapping frames, but for the rest of the workflow the indices are reversed
        once more (turning them incremental)

        Args:
            _indices: list(int)

        Returns:
            clip: dict. The format should be consistent with that of the batch input for 
                training pass, i.e., inputs argument in `DynamiteModel.forward()`
                The dictionary contains the following keys:
                * images: (T,3,H,W) RGB image tensors
                * num_instances_per_frame: list specifying #instances in each frame
                * num_clicks_per_object: #clicks sampled on each foreground object, in each frame
                * max_timestamp_list: timestamp of the latest click sampled in each frame
                * fg_coords_list: list of foreground clicks
                * bg_coords_list: list of background clicks
                * seq_name: name of the parent sequence
                * frame_indices: global indices of the clip (w.r.t. the whole sequence)
        """
        indices = _indices
        if indices[1]<indices[0]:
            indices = _indices[::-1]
        # indices - always incremental
        # _indices - true order
        
        clip = {
                "seq_name": self.sequence_id,
                "frame_indices": indices,
                "images": torch.as_tensor(self.images[indices[0]:indices[-1]+1], dtype=torch.uint8),
                "instance_ids": self.instances,
                "instances_per_frame": self.instances_per_frame[indices[0]:indices[-1]+1],
                "num_clicks_per_object": self.num_clicks_per_object[indices[0]:indices[-1]+1],
                "max_timestamp_list": self.max_timestamps[indices[0]:indices[-1]+1],
        }

        # in case no foreground clicks are found on a given instance, simulate 
        # some from the predicted masks of the overlapping frames
        net_clicks = np.sum(clip["num_clicks_per_object"], axis=0)
        instances_in_clip = np.asarray(list(set(item for sublist in self.instances_per_frame[indices[0]:indices[-1]+1] for item in sublist))) - 1
        
        # TODO: empty clip
        # TODO: before an instance appears
        if not all(net_clicks[instances_in_clip]):
        # if not all(np.sum(clip["num_clicks_per_object"], axis=0)):
            
            overlapping_frame_indices = sorted(_indices[:self.num_overlapping_frames])
            overlapping_frame_preds = torch.stack(self.pred_masks[overlapping_frame_indices[0]:overlapping_frame_indices[-1]+1])
            t = max(clip["max_timestamp_list"]) + 1

            click_counts = np.sum(clip["num_clicks_per_object"], axis=0)
            for inst_id, cc in enumerate(click_counts):
                # only sample for instances that didn't get a click
                if cc>0:
                    continue
                inst_masks = overlapping_frame_preds[:,inst_id]
                choice_range = list(range(inst_masks.shape[0]))
                while True:
                    if len(choice_range) == 0:
                        # for at least one instance, no prediction was found in the overlapping frames
                        # obtain a click from ground truth mask, if the latter is non-empty
                        gt_mask = self.gt_binary_masks[overlapping_frame_indices[0]][inst_id]
                        if gt_mask.any():
                            center_coords = get_center_coords(gt_mask)
                            self.fg_coords_list[overlapping_frame_indices[0]][inst_id].append([center_coords[0], center_coords[1], inst_id, overlapping_frame_indices[0], t])
                            # self.num_clicks_per_object[overlapping_frame_indices[0]][inst_id] += 1
                            self.max_timestamps[overlapping_frame_indices[0]] = t
                        break
                    
                    # randomly select one of the overlapping frames to sample a foreground click from
                    choice = random.sample(choice_range, 1)[0]
                    choice_range.remove(choice)
                    if inst_masks[choice].any():
                        # obtain a click from the predicted mask area
                        center_coords = get_center_coords(inst_masks[choice])
                        fr_idx = overlapping_frame_indices[choice]
                        self.fg_coords_list[fr_idx][inst_id].append([center_coords[0], center_coords[1], inst_id, fr_idx, t])
                        # self.num_clicks_per_object[fr_idx][inst_id] += 1
                        self.max_timestamps[fr_idx] = t
                        t += 1
                        break
            clip["num_clicks_per_object"] = self.num_clicks_per_object[indices[0]:indices[-1]+1]
            clip["max_timestamp_list"] = self.max_timestamps[indices[0]:indices[-1]+1]
        
        clip["fg_coords_list"] = self.fg_coords_list[indices[0]:indices[-1]+1]
        clip["bg_coords_list"] = self.bg_coords_list[indices[0]:indices[-1]+1]
        
        return clip
    

    def store_pred_masks(self, pred_masks, indices=None):
        """
        Store predicted masks of a clip

        Args:
            pred_masks: predicted masks, list of [T,H,W] tensors where T=length of the clip
            indices: list of indices (w.r.t. the whole sequence) specifying the clip
        """
        if indices is None:
            self.pred_masks = pred_masks
            return
        
        # add empty masks for instances that were not present in this clip
        instance_id_to_index = {id_: i for i, id_ in enumerate(self.instances)}
        N = len(self.instances)
        H, W = pred_masks[0].shape[1:]
        
        for idx, pred in zip(indices, pred_masks):
            updated_fr_pred = torch.zeros((N, H, W))
            # instances present in the current frame
            for i, orig_id in enumerate(self.instances_per_frame[idx]):
                final_index = instance_id_to_index[orig_id]
                updated_fr_pred[final_index] = pred[i]
            
            self.pred_masks[idx] = updated_fr_pred.to(dtype=pred.dtype, device=pred.device)

    
    def store_predicted_semantic_maps(self, indices=None):
        """
        Convert model-returned instance binary masks to semantic maps
        """
        if len(self.instances) == 1:
            self.pred_semantic_maps = [m.squeeze(0).numpy() for m in self.pred_masks]
            return
        
        out_masks_ = []
        H,W = self.pred_masks[0].shape[-2:]

        if indices is None:
            pred_masks = self.pred_masks
        else:
            pred_masks = self.pred_masks[indices[0]:indices[-1]+1]

        for idx in range(len(pred_masks)):
            dummy_ = np.zeros((H,W))
            for k in self.instances:
                dummy_ += pred_masks[idx][k-1].numpy() * k
            out_masks_.append(dummy_)
        
        if indices is None:
            self.pred_semantic_maps = np.stack(out_masks_, axis=0)
        else:
            for i, idx in enumerate(range(indices[0], indices[-1] + 1)):
                self.pred_semantic_maps[idx] = out_masks_[i]

        
    def save_visualization(self, vis_path, round_num=0, indices=None):
        """
        Save predicted mask visualization to the disc

        Args:
            vis_path: str, path to the directory where visualizations are to be saved
            round_num: int, current round number
            indices: (optional) list, save visualizations for specified indices # TODO
        """
        vis_path = os.path.join(vis_path, self.sequence_id)
        os.makedirs(vis_path, exist_ok=True)
        
        vis_path = os.path.join(vis_path, str(round_num))
        if os.path.isdir(vis_path):
            print(f"Warning! Overwriting some files in {vis_path}") # TODO - use logger warning
        else:
            os.makedirs(vis_path)
        
        for fr_idx, fr_msk in enumerate(self.pred_semantic_maps):
            m = Image.fromarray(fr_msk.astype(np.uint8))
            m.putpalette(davis_palette)
            m.save(os.path.join(vis_path, f"mask_{fr_idx}.png"))