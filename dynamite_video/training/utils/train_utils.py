import math
import torch
import numpy as np
import copy
import cv2
import random

def compute_iou(gt_masks, pred_masks, max_objs=15, iou_thres = 0.90):

    intersections = np.sum(np.logical_and(gt_masks, pred_masks), (1,2))
    unions = np.sum(np.logical_or(gt_masks,pred_masks), (1,2))
    if not unions.all():
        # at least for one of the instances, there's no gt mask and no pred mask
        # that's a correct prediction
        pos = np.where(unions==0)
        unions[pos] = 1
        intersections[pos] = 1
    
    ious = intersections/unions

    indices = torch.topk(torch.tensor(ious), len(ious),largest=False).indices
    worst_indexs = []
    i=0
    while(i<max_objs and i<len(indices)):
        if ious[indices[i]] < iou_thres:
            worst_indexs.append(indices[i])
        i+=1
        if len(worst_indexs)==max_objs:
            break
    return worst_indexs

def get_next_clicks(data, pred_output, timestamp, batched_num_clicks_per_object=None,
                       fg_coords=None, bg_coords = None,
                       max_timestamp = None
):
    
    # OPTIMIZATION
    # directly take data as input as they are already on the device
    gt_masks_batch = [x for x in data["instance_masks"]]
    pred_masks_batch = [x.cpu().numpy() for x in pred_output]
    semantic_maps_batch = [x for x in data['semantic_masks']]

    padding_mask = data["padding_mask"]
    
    for i, (gt_masks_per_image, pred_masks_per_image, semantic_map) in enumerate(zip(gt_masks_batch, pred_masks_batch, semantic_maps_batch)):
        
        # id of the instance to be refined
        indices = compute_iou(gt_masks_per_image,pred_masks_per_image)
        # if unique_timestamp:
        timestamp = max(max_timestamp)+1
        # if scribbles:
        for j in indices:
            sampled_coords_info = _get_corrective_clicks(pred_masks_per_image[j], gt_masks_per_image[j],
                                                        semantic_map, padding_mask, timestamp = timestamp,
                                                        fr_idx=i, inst_id=int(j), max_num_points=2)
            
            if sampled_coords_info is not None:
                point_coords, obj_indices = sampled_coords_info
                # if unique_timestamp:
                timestamp += len(point_coords)
                for k, obj_indx in enumerate(obj_indices):
                    if obj_indx == -1:
                        if bg_coords[i]:
                            bg_coords[i].extend([point_coords[k]])
                        else:
                            bg_coords[i] = [point_coords[k]]
                    else:
                        fg_coords[i][obj_indx].extend([point_coords[k]])
                        batched_num_clicks_per_object[i][obj_indx]+= 1
        # if unique_timestamp:
        max_timestamp[i] = timestamp-1

        return batched_num_clicks_per_object,  fg_coords, bg_coords, max_timestamp
                  
def _get_corrective_clicks(pred_mask, gt_mask, semantic_map, padding_mask,
                           timestamp, fr_idx, inst_id, max_num_points=2,
):
    gt_mask = np.asarray(gt_mask, dtype = np.bool_)
    pred_mask = np.asarray(pred_mask, dtype = np.bool_)
    padding_mask = np.asarray(padding_mask, dtype = np.bool_)

    fn_mask =  np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_mask =  np.logical_and(np.logical_not(gt_mask), pred_mask)
    
    fn_mask = np.logical_and(fn_mask, np.logical_not(padding_mask))
    fp_mask = np.logical_and(fp_mask, np.logical_not(padding_mask))
   
    H, W = gt_mask.shape

    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
    fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
    fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    if fn_max_dist > fp_max_dist:
        inner_mask = fn_mask_dt > (fn_max_dist / 2.0)
    else:
        inner_mask = fp_mask_dt > (fp_max_dist / 2.0)

    sample_locations = np.argwhere(inner_mask)
    if len(sample_locations) > 0:
        _probs = [0.80,0.20]
        num_points = 1+ np.random.choice(np.arange(max_num_points), p=_probs)
        num_points = min(num_points, sample_locations.shape[0])
        
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        H, W = pred_mask.shape
        points_coords = []
        obj_indices = []
        for index in indices:
            coords = sample_locations[index]
        
            points_coords.append([coords[0], coords[1],inst_id, fr_idx, timestamp])
            obj_indx = semantic_map[coords[0]][coords[1]] -1
            obj_indices.append(obj_indx)
            timestamp+=1
        return (points_coords, obj_indices)
    else:
        None


def get_spatiotemporal_embeddings(pos_tensor, positional_embeddings, hidden_dim=256):
        
        scale = 2 * math.pi
        if positional_embeddings == "temporal":
            dim_t = torch.arange(hidden_dim, dtype=torch.float, device=pos_tensor.device)
            dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / hidden_dim)
            t_embed = pos_tensor[:, 2] * scale
            pos_t = t_embed[:, None] / dim_t
            pos_t[:, 0::2][torch.where(pos_t[:, 0::2] < 0)] = 0.0
            pos_t[:, 1::2][torch.where(pos_t[:, 1::2] < 0)] = math.pi/2
            pos_t = torch.stack((pos_t[:, 0::2].sin(), pos_t[:, 1::2].cos()), dim=2).flatten(1)
            return pos_t
        hidden_dim = hidden_dim // 2    # 256-> 128
        dim_t = torch.arange(hidden_dim, dtype=torch.float, device=pos_tensor.device)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / hidden_dim)
        x_embed = pos_tensor[:, 1] * scale
        y_embed = pos_tensor[:, 0] * scale
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x[:, 0::2][torch.where(pos_x[:, 0::2] < 0)] = 0.0
        pos_x[:, 1::2][torch.where(pos_x[:, 1::2] < 0)] = math.pi/2
        pos_y[:, 0::2][torch.where(pos_y[:, 0::2] < 0)] = 0.0
        pos_y[:, 1::2][torch.where(pos_y[:, 1::2] < 0)] = math.pi/2
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)

        if positional_embeddings == "spatial":
            return torch.cat((pos_y, pos_x), dim=1)
        elif positional_embeddings == "spatio_temporal":
            t_embed = pos_tensor[:, 2] * scale
            pos_t = t_embed[:, None] / dim_t
            pos_t[:, 0::2][torch.where(pos_t[:, 0::2] < 0)] = 0.0
            pos_t[:, 1::2][torch.where(pos_t[:, 1::2] < 0)] = math.pi/2
            pos_t = torch.stack((pos_t[:, 0::2].sin(), pos_t[:, 1::2].cos()), dim=2).flatten(1)
            
            pos = torch.cat((pos_y, pos_x, pos_t), dim=1)
            return pos
        

 