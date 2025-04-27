import cv2
import numpy as np
import random
import torch

from functools import lru_cache


def compute_iou(
    gt_masks,
    pred_masks,
    max_insts_to_refine=15,
    iou_thres=0.90
):
    """
    Given the ground truth masks and prediction masks, compute instances-wise IoU and return the 
    indices of the worst segmented instances

    Args:
        gt_masks: ground truth masks [T, N, H, W]
        pred_masks: prediction masks [T, N, H, W]
        max_insts_to_refine: sample corrective clicks on upto this many objects (int, default=15)
        iou_thres: refine an object if computed IoU is lower than this threshold (float, default=0.90)
    
    Returns:
        worst_indices: indices of objects to refine
    """

    intersections = np.sum(np.logical_and(gt_masks, pred_masks), (1,2))
    unions = np.sum(np.logical_or(gt_masks,pred_masks), (1,2))
    
    # some instance(s) may be absent in some frame(s) in the clip. In such cases, intersection is 0,
    # regardless of the prediction. However, if the union is 0, that means the prediction was correct
    # (and empty, just like prediction). In that case, IoU should be 1.
    if not unions.all():
        # at least for one of the instances, there's no gt mask and no pred mask that's a correct prediction
        pos = np.where(unions==0)
        unions[pos] = 1
        intersections[pos] = 1

    ious = intersections/unions

    # identify the instances with worst IoUs
    indices = torch.topk(torch.tensor(ious), len(ious),largest=False).indices
    worst_indices = []
    for inst_id in indices:
        if ious[inst_id] < iou_thres:
            worst_indices.append(inst_id)
        if len(worst_indices)==max_insts_to_refine:
            break
    return worst_indices


def get_next_clicks(
    data,
    pred_output,
    num_clicks_per_object,
    fg_coords,
    bg_coords,
    max_timestamp,
    max_num_points=2
):
    """
    Given the predicted masks of current round, sample corrective clicks

    Args:
        data: dataloader input
        pred_output: predicted masks, [T, N, H, W]
        num_clicks_per_object: list of click counts on each instance, in each frame of the clip
        fg_coords: list of fg clicks on the frames of the clip
        bg_coords: list bg clicks on the frames of the clip
        max_timestamp: list of timestamps of the last clip on each frame of the clip
        max_num_points: maximum number of corrective clicks to sample per instance (int, default=2)

    Returns:
        num_clicks_per_object, fg_coords, bg_coords, max_timestamp: updated with sampled clicks
    """

    # directly take data as input as they are already on the device
    gt_masks_clip = [x.cpu().numpy() for x in data["instance_masks"]]      # [T,N,H,W]
    pred_masks_clip = [x.cpu().numpy() for x in pred_output]               # [T,N,H,W]
    semantic_maps_clip = [x.cpu().numpy() for x in data['semantic_masks']] # [T,H,W]
    padding_mask = data["padding_mask"].cpu().numpy()                      # [H,W]
    
    for fr_idx, (gt_masks, pred_masks, semantic_map) in enumerate(zip(gt_masks_clip, pred_masks_clip, semantic_maps_clip)):
        
        # id of the instance to be refined
        indices = compute_iou(gt_masks, pred_masks)
        # timestamp of the latest click so far
        timestamp = max(max_timestamp)

        for inst_id in indices:
            sampled_clicks = _get_corrective_clicks(pred_masks[inst_id], gt_masks[inst_id], semantic_map, padding_mask,
                                                        timestamp+1, max_num_points)
            
            if sampled_clicks is not None:
                for click in sampled_clicks:
                    # click is in the format [y,x,i,t]
                    click_y, click_x, click_obj, click_time = click
                    
                    # BG click
                    if click_obj == -1:
                        if bg_coords[fr_idx]:
                            bg_coords[fr_idx].extend([click_y, click_x, click_obj, fr_idx, click_time])
                        else:
                            bg_coords[fr_idx] = [click_y, click_x, click_obj, fr_idx, click_time]
                    
                    # FG click
                    else:
                        fg_coords[fr_idx][click_obj].extend([click_y, click_x, click_obj, fr_idx, click_time])
                        num_clicks_per_object[fr_idx][click_obj]+= 1

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
    last_value = 1
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
    padding_mask,
    timestamp,
    max_num_points=2
):
    """
    Sample corrective click on an instance, in a frame

    Args:
        pred_mask: H,W predicted segmentation mask of the instance
        gt_mask: H,W ground truth segmentation mask of the instance
        semantic_map: H,W ground truth semantic map of the frame
        padding_mask: H,W padding mask applied during data-loading
        timestamp: timestamp of current click (int)
        max_num_points: maximum #clicks to sample (int, default: 2)
    """

    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    padding_mask = np.asarray(padding_mask, dtype = np.bool_)

    # negative error map - g.t. foreground missed by the prediction
    fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
    fn_mask = np.logical_and(fn_mask, np.logical_not(padding_mask))
    
    # positive error map - g.t. background covered by the prediction
    fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
    fp_mask = np.logical_and(fp_mask, np.logical_not(padding_mask))
    
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

            # ID of the instance in the g.t. mask at the sampled click location
            obj_indx = semantic_map[coords[0]][coords[1]] - 1
            points_coords.append([coords[0], coords[1], obj_indx, timestamp])   # [y,x,i,t]
            timestamp+=1

    return points_coords