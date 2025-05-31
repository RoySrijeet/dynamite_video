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
    Find object center from binary mask

    Args:
        mask: binary mask [H, W], np.ndarray
        k: distance threshold around the center
    """
    assert mask.ndim==2

    if torch.is_tensor(mask):
        mask = mask.numpy()
    mask = mask.astype(np.uint8)

    # find distance transform - distance of each object pixel from nearest object boundary
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]

    # center region of the object
    candidates = np.argwhere(dt > (dt.max()/k))
    
    # select a point in the center region randomly
    indices = np.random.randint(0,candidates.shape[0])
    return candidates[indices]


def get_foreground_clicks(
        frame_idx,
        object_ids,
        max_class_id,
        binary_masks,
        key_objects,
        optional_frames_fg_prob=0.5,
        max_num_points=6,
        gamma=0.7,
        t=1,
):
    """
    Sample foreground clicks from a frame.

    Key objects specify the object IDs for which clicks must be sampled from this frame.
    If an object has ID > max_class_id, it is an object and one click is sampled at 
    its center. If object ID <= max_class_id, the object is a semantic class, and clicks
    are sampled randomly from the foreground area.

    For objects that appear in the frame, but are not key objects - clicks are sampled
    with a probability specified by `optional_frames_fg_prob`. All clicks sampled in this
    way are chosen randomly from foreground area.

    Args:
        frame_idx: frame index
        object_ids: IDs of the object present in the clip
        max_class_id: maximum ID of stuff classes
        binary_masks: [N, H, W] binary masks of the objects in this frame
        key_objects: sample at least one click on each of the key objects
        optional_frames_fg_prob: optional sampling probability of non-key objects
        max_num_points: maximum number of points to sample from each object in any frame
        gamma: probability scaling factor of sampling n no. of clicks
        t: starting time stamp

    Returns:
        A list of lists. A sub-list consists of the foregound clicks sampled from
        an object in the frame
    """

    assert binary_masks.ndim == 3

    fg_coords_list = []
    # object_ids are serial and 1-indexed
    num_clicks_per_object_fr = np.zeros(len(object_ids)).astype('int')
    
    count = 0
    
    for inst_id, _mask in zip(object_ids, binary_masks):
        coords = []
        
        if not _mask.any():
            # if the mask is empty, no fg click can be sampled
            fg_coords_list.append(coords)
            continue

        if inst_id not in key_objects:
            # sample only with some probability
            if np.random.rand() > optional_frames_fg_prob:
                fg_coords_list.append(coords)
                continue
        
        if inst_id > max_class_id and inst_id in key_objects:
            # object is a key object, fetch center coordinates
            center_coords = get_center_coords(_mask)
            coords.append([center_coords[0], center_coords[1], inst_id, frame_idx, t])
            num_clicks_per_object_fr[inst_id-1] += 1
            count+=1
            t+=1
            
        
        # erode mask area to avoid sampling clicks too close to object boundary
        kernel = np.ones((3,3),np.uint8)
        _eroded_m = cv2.erode(_mask, kernel, iterations=1)
        sample_locations = np.argwhere(_eroded_m)

        if sample_locations.shape[0] <= 64:
            # the object is super small, just sample from the original mask
            # NOTE: In DynaMITe, objects with area smaller than 400 are filtered out.
            # Applying a 3x3 erosion on a mask area of 400 erodes it down to 64.
            sample_locations = np.argwhere(_mask)

        # how many points to sample is determined by the probabilities
        pos_click_probs = _generate_probs(max_num_points, gamma=gamma)
        num_points = np.random.choice(np.arange(1,max_num_points+1), p=pos_click_probs)
        
        # in case there's not as many positive mask locations as the number of clicks to be sampled
        num_points = min(num_points,sample_locations.shape[0]//2)

        # randomly sample clicks
        indices = random.sample(range(sample_locations.shape[0]), num_points)
        for index in indices:
            point_coords = sample_locations[index]
            # record click
            coords.append([point_coords[0], point_coords[1], inst_id, frame_idx, t])
            num_clicks_per_object_fr[inst_id-1] += 1
            count+=1
            t+=1

        fg_coords_list.append(coords)
    return num_clicks_per_object_fr, fg_coords_list, t, count

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
        max_num_points: maximum number of points to sample from each object in any frame
        gamma: probability scaling factor of sampling n no. of clicks
        t: starting time stamp
    
    Returns:
        A list of background clicks sampled from the frame
    """
    # erode to avoid sampling clicks too close to the boundary
    kernel = np.ones((7,7),np.uint8)
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
        object_ids,
        object_masks,     # [N, T, H, W]
        bg_masks,           # [T, H, W]
        frame_object_occupancy,
        max_class_id,
        max_num_points=6, 
        optional_frames_fg_prob=0.5,
        bg_prob=0.2,
        gamma=0.7,
        start_t=1,
):
    """
    Sample clicks from the frames of a video clip. Each click is stored in the following format: 
    [y,x,i,f,t], where:
        y,x: spatial coordinates
        i: object ID at the location in g.t. mask
        f: frame index
        t: timestamp

    Args:
        object_ids: list of IDs of the objects present in the clip
        object_masks: [T, N, H, W] np.ndarray
        bg_masks: [T, H, W] np.ndarray
        frame_object_occupancy: mapping {object_id: [frames it appears in]}
        max_class_id: maximum ID of stuff classes
        max_num_points: maximum number of points to sample from each object in any frame
        optional_frames_fg_prob: probability of sampling fg clicks on more frames
        bg_prob: probability of sampling bg clicks on any given frame
        gamma: probability scaling factor of sampling n no. of clicks
        start_t: starting time stamp for each clip, (default: 1)
    """ 
    assert object_ids == sorted(frame_object_occupancy.keys())  # sanity check
    # no. of frames in the clip
    num_frames = bg_masks.shape[0]
    
    # how many clicks each object receives in each frame
    num_clicks_per_object = np.zeros((num_frames, len(object_ids))).astype('int')
    # timestamp of the latest click on each frame
    max_timestamp_clip = [0] * num_frames

    # for each object, randomly a select a frame (wherein it appears) and sample clicks in the 
    # selected frame for this specific object. This ensures that each object receives a click
    sample_object_from = defaultdict(list)
    for obj_id, fr_idxs in frame_object_occupancy.items():
        choice = random.choice(fr_idxs)
        # add a click on object `inst_id` in frame `choice`
        sample_object_from[choice].append(obj_id)
    
    # all clicks in a clip share a single timeline
    t = start_t
    
    fg_coords_list = []
    for fr_idx in range(num_frames):

        key_objects = []
        if fr_idx in sample_object_from.keys():
            # there must be clicks sampled on these objects in this frame
            key_objects = sample_object_from[fr_idx]

        num_clicks_per_object_fr, fg_coords_list_fr, t, count = get_foreground_clicks(fr_idx,
                                                                                    object_ids,
                                                                                    max_class_id,
                                                                                    object_masks[fr_idx],
                                                                                    key_objects,
                                                                                    optional_frames_fg_prob,
                                                                                    max_num_points,
                                                                                    gamma,
                                                                                    t,
                                                                                )
        # update click records
        num_clicks_per_object[fr_idx] += num_clicks_per_object_fr
        fg_coords_list.append(fg_coords_list_fr)
        if count > 0:
            max_timestamp_clip[fr_idx]=t-1

    # BG clicks
    bg_coords_list = []
    for fr_idx in range(num_frames):

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
        bg_coords_list.append(bg_coords_list_fr)
        if len(bg_coords_list_fr) > 0:
            max_timestamp_clip[fr_idx]=t-1


    return num_clicks_per_object.tolist(), fg_coords_list, bg_coords_list, max_timestamp_clip