import cv2
import numpy as np
import random
import torch

from functools import lru_cache


def compute_iou(
    gt_masks,
    pred_masks,
    strategy,
    max_objects_to_refine=15,
    iou_thres=0.95
):
    """
    Given the ground truth masks and prediction masks, compute object-wise IoU and return the 
    indices of the worst segmented objects

    Args:
        gt_masks: ground truth masks [T, N, H, W]
        pred_masks: prediction masks [T, N, H, W]
        max_objects_to_refine: sample corrective clicks on upto this many objects (int, default=15)sque
        iou_thres: refine an object if computed IoU is lower than this threshold (float, default=0.90)
    
    Returns: indices of up to `max_objects_to_refine` objects that have the lowest average IoU
    """

    intersections = torch.sum(torch.logical_and(gt_masks, pred_masks), (2,3))
    unions = torch.sum(torch.logical_or(gt_masks, pred_masks), (2,3))

    iou = torch.where(unions == 0,
                        torch.ones_like(unions),              # if union == 0, set iou = 1
                        intersections.float() / unions.float()
                )
    avg_iou = torch.mean(iou, 0)
    max_objects_to_refine = np.random.randint(1, min(len(avg_iou), max_objects_to_refine)+1)
    
    if strategy == "worst":
        values, indices = torch.topk(avg_iou, k=max_objects_to_refine, largest=False)
        values = values[values < iou_thres]
        if len(values) < 1:
            return None, None
        
        worst_objects = indices[:len(values)]
        worst_frames = torch.argmin(iou[:, worst_objects], dim=0)
        return worst_objects, worst_frames
    
    if strategy == "random":
        mask = iou < iou_thres
        valid_indices = torch.nonzero(mask, as_tuple=False)
        num_valid = valid_indices.shape[0]
        if num_valid < 1:
            return None, None
        
        perm = torch.randperm(num_valid)
        selected_pairs = valid_indices[perm[:min(max_objects_to_refine, num_valid)]]
        return selected_pairs[:, 1], selected_pairs[:, 0]


def get_next_clicks(
    data,
    pred_output,
    num_clicks_per_object,
    fg_coords,
    bg_coords,
    max_timestamp,
    max_objects_to_refine,
    iou_threshold,
    refine_strategy
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

    assert max_objects_to_refine >= 1
    
    clip_gt_masks = data["binary_masks"].detach().cpu()
    clip_pred_masks = torch.stack(pred_output).detach().cpu()

    refine_objects, refine_frames = compute_iou(clip_gt_masks, clip_pred_masks, 
                                strategy=refine_strategy, max_objects_to_refine=max_objects_to_refine, iou_thres=iou_threshold)
    
    if refine_objects is None:
        return num_clicks_per_object, fg_coords, bg_coords, max_timestamp
    
    clip_gt_masks = clip_gt_masks.numpy()
    clip_pred_masks = clip_pred_masks.numpy()
    clip_panoptic_masks = data['panoptic_masks'].detach().cpu().numpy()
    padding_mask = np.logical_not(data['padding_mask'].cpu().numpy())

    count = 0
    for obj_id, fr_idx in zip(refine_objects, refine_frames):
        gt_masks = clip_gt_masks[fr_idx] * padding_mask
        pred_masks = clip_pred_masks[fr_idx]
        panoptic_map = clip_panoptic_masks[fr_idx]

        # timestamp of the latest click so far
        timestamp = max(max_timestamp)
        sampled_clicks = _get_corrective_clicks(pred_masks[obj_id], gt_masks[obj_id], panoptic_map,
                                                    timestamp+1, max_num_points=1)
        
        if sampled_clicks is not None:
            for click in sampled_clicks:
                # click is in the format [y,x,i,t]
                click_y, click_x, click_obj, click_time = click
                
                # BG click
                if click_obj == -1:
                    bg_coords.append([click_y, click_x, click_obj, fr_idx, click_time])
                
                # FG click
                else:
                    total_num_clicks_per_obj = np.asarray(num_clicks_per_object).sum(axis=0)
                    insert_idx = total_num_clicks_per_obj[:click_obj].sum()
                    
                    fg_coords.insert(insert_idx+1, [click_y, click_x, click_obj+1, fr_idx.item(), click_time])
                    num_clicks_per_object[fr_idx][click_obj]+= 1

                max_timestamp[fr_idx] = click_time
                timestamp = click_time
        count += 1
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
    panoptic_map,
    timestamp,
    max_num_points=2
):
    """
    Sample corrective click on an object, in a frame

    Args:
        pred_mask: H,W predicted segmentation mask of the object
        gt_mask: H,W ground truth segmentation mask of the object
        panoptic_map: H,W ground truth panoptic map of the frame
        padding_mask: H,W padding mask applied during data-loading
        timestamp: timestamp of current click (int)
        max_num_points: maximum #clicks to sample (int, default: 2)
    """
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    
    # false negative error map - should have been part of the prediction, but is not
    fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
    
    # false positive error map - should not have been part of the prediction, but is
    fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)

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
            obj_indx = panoptic_map[coords[0]][coords[1]] - 1
            points_coords.append([coords[0], coords[1], obj_indx, timestamp])   # [y,x,i,t]
            timestamp+=1
            
    return points_coords


# from torch.nn import functional as F

# def compute_dilated_attention_mask(outputs_mask, attn_mask_target_size):
    
#     # for learnable queries, initially there is no masking
#     mask_logits = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)   # TxQxhxw
    
#     kernels = []
#     dilation_kernel_size = 7
#     kernel_size = []
#     kernels = []
#     for i in range(1, T):
#         ks = dilation_kernel_size + 4**i
#         kernels.append(torch.ones((1, 1, ks, ks), dtype=torch.float32))
#         kernel_size.append(ks)

#     T, Q, H, W = mask_logits.shape 

#     mask_logits = mask_logits.detach()

#     max_probs = mask_logits.sigmoid().view(T,Q, -1).max(dim=-1).values
#     strongest_frames = max_probs.argmax(dim=0)

#     attention_mask = torch.zeros_like(mask_logits, dtype=torch.bool)    # T,Q,H,W

#     for q in range(Q):
#         t_star = strongest_frames[q].item()
#         mask_logit = mask_logits[t_star, q]  # [H, W]

#         q_mask = (mask_logit.sigmoid() > 0.5).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

#         for t in range(T):
#             t_dist = abs(t_star - t)
#             if t_dist > 0:
#                 dilated = F.conv2d(q_mask, kernels[t_dist - 1], padding=kernel_size[t_dist-1]//2)
#                 dilated = (dilated > 0).squeeze(0).squeeze(0)  # [H, W], bool
#             else:
#                 dilated = q_mask.squeeze(0).squeeze(0)  # [H, W], bool

#             attention_mask[t,q] = ~dilated.bool()
    
#     return attention_mask