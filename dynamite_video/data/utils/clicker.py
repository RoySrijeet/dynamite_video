import cv2
import numpy as np
import random
import torch

from collections import defaultdict
from functools import lru_cache


@lru_cache(maxsize=None)
def _generate_probs(max_num_points, gamma):
    """
    Sampling probability of n-th click.
    If n-th click has prob p, (n+1)th click has prob p*gamma.

    Args:
        max_num_points: max no. of points to sample
        gamma: probability scaling factor
    """
    probs = []
    last_value = 1.
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs)
    probs /= probs.sum()

    return probs


def get_center_coords(mask, k=1.7):
    """
    Find target center from binary mask

    Args:
        mask: binary mask [H, W], np.ndarray
        k: distance threshold around the center
    """
    assert mask.ndim==2

    if torch.is_tensor(mask):
        mask = mask.numpy()
    mask = mask.astype(np.uint8)

    # find distance transform - distance of each pixel from nearest target boundary
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    # center region of the target
    candidates = np.argwhere(dt > (dt.max()/k))
    
    # select a point in the center region randomly
    indices = np.random.randint(0,candidates.shape[0])
    return candidates[indices]


def get_foreground_clicks(
        obj_id,
        obj_mask,
        max_timestamp,
        optional_frames_fg_prob,
        max_num_points=6,
        gamma=0.7,
        t=1,
):
    """
    obj_id: int, object id
    obj_mask: np.ndarray of shape T,H,W; binary masks of the object in the frames
    optional_frames_fg_prob: optional sampling probability of non-key targets
    max_num_points: maximum number of points to sample from each target in any frame
    gamma: probability scaling factor of sampling n no. of clicks
    t: starting time stamp
    """
    assert obj_mask.ndim == 3, f"Expecting object mask of shape (T,H,W)!"
    
    T = obj_mask.shape[0]
    
    # num of clicks sampled for this object across all frames
    clicks_count_per_frame = np.zeros((T), dtype=np.uint16)
    # clicks sampled on the object
    clicks_on_obj = []
    
    # frames where the target object appears
    available_frames = np.nonzero(obj_mask.any((1,2)))[0].tolist()
    # randomly select a frame to sample a center click from
    fr_idx = random.choice(available_frames)

    # get center coordinates
    center_coords = get_center_coords(obj_mask[fr_idx], k=1.1)
    # record click
    clicks_on_obj.append([center_coords[0], center_coords[1], obj_id, fr_idx, t])
    clicks_count_per_frame[fr_idx] += 1
    max_timestamp[fr_idx] = t
    t += 1
    
    max_num_points -= 1

    if np.random.rand() > optional_frames_fg_prob or max_num_points == 0:
        return clicks_count_per_frame, clicks_on_obj, max_timestamp, t

    # sample extra clicks on other frames
    
    # how many extra points to sample is determined by the probabilities
    pos_click_probs = _generate_probs(max_num_points, gamma=gamma)
    num_points = np.random.choice(np.arange(1,max_num_points+1), p=pos_click_probs)
    kernel = np.ones((3,3),np.uint8)
    
    for i in range(num_points):
        # randomly select a frame
        fr_idx = random.choice(available_frames)
        
        # erode mask area to avoid sampling clicks too close to target boundary
        _eroded_m = cv2.erode(obj_mask[fr_idx].copy(), kernel, iterations=1)
        sample_locations = np.argwhere(_eroded_m)
        if sample_locations.shape[0] <= 64:
            # the target is super small, just sample from the original mask
            # Applying a 3x3 erosion on a mask area of 400 erodes it down to 64.
            sample_locations = np.argwhere(obj_mask[fr_idx])
        
        # randomly select a location
        index = random.sample(range(sample_locations.shape[0]), 1)
        extra_coords = sample_locations[index]
        # record click
        clicks_on_obj.append([extra_coords[0], extra_coords[1], obj_id, fr_idx, t])
        clicks_count_per_frame[fr_idx] += 1
        max_timestamp[fr_idx] = t
        t+=1

    return clicks_count_per_frame, clicks_on_obj, max_timestamp, t
    

def get_background_clicks(
        frame_idx,
        bg_mask,
        max_num_points=2,
        gamma=0.7,
        t=1,
):
    """
    Sample background clicks from the binary background mask of a frame

    Args:
        frame_idx: frame index
        bg_mask: binary background mask of shape [H, W]
        max_num_points: maximum number of points to sample from each target in any frame
        gamma: probability scaling factor of sampling n no. of clicks
        t: starting time stamp
    
    Returns:
        A list of background clicks sampled from the frame
    """
    # erode to avoid sampling clicks too close to the boundary
    kernel = np.ones((3,3),np.uint8)
    _eroded_bg_mask = cv2.erode(bg_mask, kernel, iterations=3)
    sample_locations = np.argwhere(_eroded_bg_mask)

    if sample_locations.shape[0] <= 64:
        sample_locations = np.argwhere(bg_mask)

    neg_click_probs = _generate_probs(max_num_points, gamma=gamma)
    num_points = np.random.choice(np.arange(1, max_num_points+1), p=neg_click_probs)
    num_points = min(num_points,sample_locations.shape[0]//2)
    indices = random.sample(range(sample_locations.shape[0]), num_points)

    coords = []
    for index in indices:
        point_coords = sample_locations[index]
        coords.append([point_coords[0], point_coords[1], -1, frame_idx, t])
        t+=1
    
    return coords, t



def get_clicks_coords(
        target_masks,     # [T, N, H, W]
        max_num_points=6, 
        optional_frames_fg_prob=0.5,
        bg_masks=None,           # [T, H, W]
        bg_prob=0.,
        gamma=0.7,
        start_t=1,
):
    """
    Sample clicks from the frames of a video clip. Each click is stored in the following format: 
    [y,x,i,f,t], where:
        y,x: spatial coordinates
        i: target ID at the location in g.t. mask
        f: frame index
        t: timestamp

    Args:
        target_masks: [T, N, H, W] np.ndarray
        max_num_points: maximum number of points to sample from each target in any frame
        optional_frames_fg_prob: probability of sampling fg clicks on more frames
        bg_masks: [T, H, W] np.ndarray
        bg_prob: probability of sampling bg clicks on any given frame
        gamma: probability scaling factor of sampling n no. of clicks
        start_t: starting time stamp for each clip, (default: 1)
    """ 
    # no. of frames in the clip
    T,N,H,W = target_masks.shape
    
    # how many clicks each target receives in each frame
    num_clicks_per_target = np.zeros((T, N)).astype(np.uint16)
    # timestamp of the latest click on each frame
    max_timestamp = [0] * T
    
    # all clicks in a clip share a single timeline
    t = start_t
    
    #### FG clicks ####
    
    fg_coords_list = []
    # sample a click on each target object
    for obj_id in range(N):
        # object mask
        obj_mask = target_masks[:,obj_id]   # T,H,W

        (clicks_count_per_frame, 
         clicks_on_obj, 
         max_timestamp, t) = get_foreground_clicks(obj_id+1,
                                                    obj_mask,
                                                    max_timestamp,
                                                    optional_frames_fg_prob,
                                                    max_num_points,
                                                    gamma,
                                                    t,
                                                )
        # update click records
        num_clicks_per_target[:,obj_id] += clicks_count_per_frame
        fg_coords_list.extend(clicks_on_obj)

    
    #### BG clicks ####
    bg_coords_list = []
    if bg_masks is not None:
        for fr_idx in range(T):

            # with some probability, sample -ve clicks from this frame
            if np.random.rand() > bg_prob:
                bg_coords_list.append([])
                continue

            bg_coords_list_fr, t = get_background_clicks(
                                                fr_idx,
                                                bg_masks[fr_idx],
                                                max_num_points,
                                                gamma,
                                                t
                                            )
            bg_coords_list.extend(bg_coords_list_fr)
            if len(bg_coords_list_fr) > 0:
                max_timestamp[fr_idx]=t-1


    return num_clicks_per_target.tolist(), fg_coords_list, bg_coords_list, max_timestamp