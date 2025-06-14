import cv2
import numpy as np
import random
import torch

from collections import defaultdict
from functools import lru_cache

def compute_iou(
    gt_masks,
    pred_masks,
    max_objects_to_refine=15,
    iou_thres=0.90
):
    """
    Given the ground truth masks and prediction masks, compute object-wise IoU and return the 
    indices of the worst segmented objects

    Args:
        gt_masks: ground truth masks [T, N, H, W]
        pred_masks: prediction masks [T, N, H, W]
        max_objects_to_refine: sample corrective clicks on upto this many objects (int, default=15)sque
        iou_thres: refine an object if computed IoU is lower than this threshold (float, default=0.90)
    
    Returns:
        worst_indices: indices of objects to refine
    """

    intersections = np.sum(np.logical_and(gt_masks, pred_masks), (1,2))
    unions = np.sum(np.logical_or(gt_masks,pred_masks), (1,2))
    
    # some object(s) may be absent in some frame(s) in the clip. In such cases, intersection is 0,
    # regardless of the prediction. However, if the union is 0, that means the prediction was correct
    # (and empty, just like prediction). In that case, IoU should be 1.
    if not unions.all():
        # at least for one of the objects, there's no gt mask and no pred mask that's a correct prediction
        pos = np.where(unions==0)
        unions[pos] = 1
        intersections[pos] = 1

    ious = intersections/unions

    # identify the objects with worst IoUs
    indices = torch.topk(torch.tensor(ious), len(ious),largest=False).indices
    worst_indices = []
    for obj_id in indices:
        if ious[obj_id] < iou_thres:
            worst_indices.append(obj_id)
        if len(worst_indices)==max_objects_to_refine:
            break
    return worst_indices


def get_next_clicks(
    data,
    pred_output,
    num_clicks_per_object,
    fg_coords,
    bg_coords,
    max_timestamp,
    num_objects_to_refine,
    max_num_points=1,
    visualize=False,
    train_iter=None,
    round_num=None
):
    """
    Given the predicted masks of current round, sample corrective clicks

    Args:
        data: dataloader input
        pred_output: predicted masks, [T, N, H, W]
        num_clicks_per_object: list of click counts on each object, in each frame of the clip
        fg_coords: list of fg clicks on the frames of the clip
        bg_coords: list bg clicks on the frames of the clip
        max_timestamp: list of timestamps of the last clip on each frame of the clip
        max_num_points: maximum number of corrective clicks to sample per object (int, default=2)

    Returns:
        num_clicks_per_object, fg_coords, bg_coords, max_timestamp: updated with sampled clicks
    """

    # directly take data as input as they are already on the device
    gt_masks_clip = [x.cpu().numpy() for x in data["binary_masks"]]        # [T,N,H,W]
    pred_masks_clip = [x.cpu().numpy() for x in pred_output]               # [T,N,H,W]
    semantic_maps_clip = [x.cpu().numpy() for x in data['semantic_masks']] # [T,H,W]
    # ignore_masks_clip = [x.cpu().numpy().astype(np.uint8) for x in data['ignore_masks']]  # [T,H,W]
    # padding_mask = data["padding_mask"].cpu().numpy()                      # [H,W]

    # if visualize:
    #     import os
    #     visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/training_clicker"
    #     torch.save(gt_masks_clip,       os.path.join(visualize_dir, f"gt_masks_round_{round_num}_iter_{train_iter}.pth"))
    #     torch.save(pred_masks_clip,     os.path.join(visualize_dir, f"pred_masks_round_{round_num}_iter_{train_iter}.pth"))
    #     torch.save(semantic_maps_clip,  os.path.join(visualize_dir, f"semantic_maps_round_{round_num}_iter_{train_iter}.pth"))
    #     torch.save(ignore_masks_clip,   os.path.join(visualize_dir, f"ignore_masks_round_{round_num}_iter_{train_iter}.pth"))
    #     torch.save(padding_mask,        os.path.join(visualize_dir, f"padding_mask_round_{round_num}_iter_{train_iter}.pth"))
    
    candidates_for_refinement = defaultdict(list)
    for fr_idx, (gt_masks, pred_masks) in enumerate(zip(gt_masks_clip, pred_masks_clip)):
        
        # id of the object to be refined
        indices = compute_iou(gt_masks, pred_masks)
        for obj_id in indices:
            candidates_for_refinement[obj_id.item()].append(fr_idx)

    num_objects_to_refine = np.random.randint(1, min(len(candidates_for_refinement), num_objects_to_refine)+1)
    # randomly select objects to be refined
    objects_to_refine = np.random.choice(list(candidates_for_refinement.keys()), num_objects_to_refine)

    for obj_id in objects_to_refine:
        # randomly select a frame to sample clicks from
        fr_idx = np.random.choice(candidates_for_refinement[obj_id])
        gt_masks = gt_masks_clip[fr_idx]
        pred_masks = pred_masks_clip[fr_idx]
        semantic_map = semantic_maps_clip[fr_idx]
        # ignore_mask = ignore_masks_clip[fr_idx]

        # timestamp of the latest click so far
        timestamp = max(max_timestamp)
        sampled_clicks = _get_corrective_clicks(pred_masks[obj_id], gt_masks[obj_id], semantic_map, 
                                                    # ignore_mask, padding_mask,
                                                    timestamp+1, max_num_points)
        
        if sampled_clicks is not None:
            for click in sampled_clicks:
                # click is in the format [y,x,i,t]
                click_y, click_x, click_obj, click_time = click
                
                # BG click
                if click_obj == -1:
                    bg_coords[fr_idx].append([click_y, click_x, click_obj, fr_idx, click_time])
                
                # FG click
                else:
                    fg_coords[fr_idx][click_obj].append([click_y, click_x, click_obj+1, fr_idx, click_time])
                    num_clicks_per_object[fr_idx][click_obj]+= 1

                max_timestamp[fr_idx] = click_time
                timestamp = click_time

    return num_clicks_per_object, fg_coords, bg_coords, max_timestamp


@lru_cache(maxsize=None)
def _generate_probs(max_num_points, gamma=0.25):
    """
    Sampling probability of n-th click.
    If n-th click has prob p, (n+1)th click has prob p*gamma.

    Args:
        max_num_points: max no. of points to sample
        gamma: probability scaling factor (float, default=0.25)
    """
    probs = []
    last_value = 1.
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma
    probs = np.array(probs)
    probs /= probs.sum()
    return probs


def _get_corrective_clicks(
    pred_mask,
    gt_mask,
    semantic_map,
    # ignore_mask,
    # padding_mask,
    timestamp,
    max_num_points=2,
):
    """
    Sample corrective click on an object, in a frame

    Args:
        pred_mask: H,W predicted segmentation mask of the object
        gt_mask: H,W ground truth segmentation mask of the object
        semantic_map: H,W ground truth semantic map of the frame
        padding_mask: H,W padding mask applied during data-loading
        timestamp: timestamp of current click (int)
        max_num_points: maximum #clicks to sample (int, default: 2)
    """
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    # padding_mask = np.asarray(padding_mask, dtype = np.bool_)
    # ignore_mask = np.asarray(ignore_mask, dtype = np.bool_)
    
    # negative error map - g.t. foreground missed by the prediction
    fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
    # fn_mask = np.logical_and(fn_mask, np.logical_not(padding_mask))
    # fn_mask = np.logical_and(fn_mask, np.logical_not(ignore_mask))
    
    # positive error map - g.t. background covered by the prediction
    fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
    # fp_mask = np.logical_and(fp_mask, np.logical_not(padding_mask))
    # fp_mask = np.logical_and(fp_mask, np.logical_not(ignore_mask))
    
    # distance transform to find the center of the error region
    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
    fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    # choose the bigger error region
    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    if fn_max_dist > fp_max_dist:
        inner_mask = fn_mask_dt > (fn_max_dist / 2.0)
    else:
        inner_mask = fp_mask_dt > (fp_max_dist / 2.0)

    # candidate click coordinates from the largest error region
    sample_locations = np.argwhere(inner_mask)
    
    points_coords = None
    if len(sample_locations) > 0:
        # num of clicks to sample
        _probs = _generate_probs(max_num_points)  #[0.80,0.20]
        num_points = 1 + np.random.choice(np.arange(max_num_points), p=_probs)
        num_points = min(num_points, sample_locations.shape[0])

        # sample from candidate click coordinates
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        
        # record the sampled click coordinates
        points_coords = []
        for index in indices:
            coords = sample_locations[index]

            # ID of the object in the g.t. mask at the sampled click location
            obj_indx = semantic_map[coords[0]][coords[1]] - 1
            points_coords.append([coords[0], coords[1], obj_indx, timestamp])   # [y,x,i,t]
            timestamp+=1

    return points_coords