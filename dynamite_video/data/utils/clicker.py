import cv2
import torch
import random
import numpy as np

from operator import add
from einops import rearrange
from functools import lru_cache
from collections import defaultdict

@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    """
    Sampling probability of n-th click.
    If n-th click has prob p, (n+1)th click has prob p*gamma.

    Args:
        max_num_points: max no. of points to sample
        gamma: probability scaling factor
    """
    probs = []
    last_value = 1
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
        instance_ids,
        binary_masks,
        key_instances,
        optional_frames_fg_prob=0.7,
        first_click_center=True,
        max_num_points=6,
        gamma=0.7,
        t=1,
):
    """
    Sample foreground clicks from the binary instance masks of a frame

    Args:
        frame_idx: frame index
        instance_ids: IDs of the instance present in the clip, in the same order
            as they appear in the binary instance masks
        binary_masks: [N, H, W] binary masks of the instances in this frame
        key_instances: sample at least one click on each of the key instances
        optional_frames_fg_prob: if there's no key instances in a frame, optionally
            sample fg clicks from this frame with this specified probability
        first_click_center: whether to sample first click at object center
        max_num_points: max no. of foreground points to sample
        gamma: probability scaling factor of sampling n no. of clicks
        t: starting time stamp

    Returns:
        A list of lists. A sub-list consists of the foregound clicks sampled from
        an instance in the frame
    """

    assert binary_masks.ndim == 3
    # sampling probs of positive clicks
    _pos_probs = generate_probs(max_num_points, gamma=gamma)

    fg_coords_list = []
    # instance_ids are serial and 1-indexed
    num_clicks_per_object_fr = np.zeros(len(instance_ids)).astype('int')
    
    count = 0
    
    for inst_id, _mask in zip(instance_ids, binary_masks):
        # for each instance present in the frame

        coords = []
        if not _mask.any():
            # if the mask is empty, no fg click
            fg_coords_list.append(coords)
            continue

        if inst_id not in key_instances:
            # if instance does not need to be sampled from this frame OR
            # if the frame is an optional frame (no key instances), then
            # sample only with some probability
            if np.random.rand() > optional_frames_fg_prob:
                fg_coords_list.append(coords)
                continue
        
        if first_click_center:
            # fetch center coordinates
            center_coords = get_center_coords(_mask)
            # record click
            coords.append([center_coords[0], center_coords[1], inst_id, frame_idx, t])
            num_clicks_per_object_fr[inst_id-1] += 1
            count+=1
            t+=1
        
        
        # sample more foreground clicks
        # erode mask area to avoid sampling clicks too close to object boundary
        kernel = np.ones((3,3),np.uint8)
        _eroded_m = cv2.erode(_mask,kernel,iterations = 1)
        sample_locations = np.argwhere(_eroded_m)

        # how many points to sample is determined by the probabilities
        num_points = np.random.choice(np.arange(max_num_points), p=_pos_probs)
        
        # in case there's not as many positive mask 
        # locations as the number of clicks to be sampled
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
        max_num_points=6,
        gamma=0.7,
        t=1,
):
    """
    Sample background clicks from the binary background mask of a frame

    Args:
        frame_idx: frame index
        bg_mask: binary background mask of shape [H, W]
        max_num_points: max no. of background points to sample
        gamma: probability scaling factor of sampling n no. of clicks
        t: starting time stamp
    
    Returns:
        A list of background clicks sampled from the frame
    """
    # sampling probs of negative clicks
    _neg_probs = generate_probs(max_num_points, gamma=gamma)

    kernel = np.ones((3,3),np.uint8)
    # erode to avoid sampling clicks too close to the boundary
    _eroded_bg_mask = cv2.erode(bg_mask, kernel,iterations = 1)
    sample_locations = np.argwhere(_eroded_bg_mask)

    num_points = np.random.choice(np.arange(max_num_points), p=_neg_probs)
    num_points = min(num_points,sample_locations.shape[0]//2)
    indices = random.sample(range(sample_locations.shape[0]), num_points)

    coords = []
    for index in indices:
        point_coords = sample_locations[index]
        coords.append([point_coords[0], point_coords[1], -1, frame_idx, t])
        t+=1
    
    return coords, t



def get_clicks_coords(
        instance_ids,       
        instance_masks,     # [N, T, H, W]
        bg_masks,           # [T, H, W]
        frame_instance_occupancy,
        max_num_points=6, 
        first_click_center=True, 
        optional_frames_fg_prob=0.7,
        bg_prob=0.2,
        gamma=0.7,
        start_t=1,
):
    """
    Add clicks randomly on the frames of a video clip

    Args:
        instance_ids: list of IDs of the instances present in the clip
        instance_masks: binary instance masks of all the frames in the video,
            used to sample foreground clicks from. [T, N, H, W] np.ndarray
        bg_masks: semantic masks of all the frames in the video, used to sample
            background clicks from. [T, H, W] np.ndarray
        frame_instance_occupancy: list of mappings. For each instance, lists which frames
            they appear in
        max_num_points: maximum number of points to sample from a *frame*
        first_click_center: whether to sample first click at object center, (default: True)
        optional_frames_fg_prob: probability of sampling fg clicks on more frames
        bg_prob: probability of sampling bg clicks on any given frame
        gamma: probability scaling factor of sampling n no. of clicks
        start_t: starting time stamp for each clip, (default: 1)
    """ 
    # no. of frames in the clip
    num_frames = bg_masks.shape[0]
    
    # how many clicks each instance receives in each frame
    # num_clicks_per_object = [[0] * len(instance_ids) for _ in range(num_frames)]
    num_clicks_per_object = np.zeros((num_frames, len(instance_ids))).astype('int')
    # timestamp of the final click on each frame
    max_timestamp_clip = [0] * num_frames

    # for each instance, randomly a select a frame (where it appears)
    # we add clicks in the selected frame on this specific instance. 
    # This ensures that each instance receives a click
    sample_instances_from = defaultdict(list)
    for inst_id, frame_idxs in frame_instance_occupancy.items():
        choice = random.choice(frame_idxs)
        # add a click on instance `inst_id` in frame `choice`
        sample_instances_from[choice].append(inst_id)
    
    # all clicks in a clip share a single timeline
    t = start_t
    fg_coords_list = []
    # Sample clicks from a given frame
    for fr_idx in range(num_frames):

        key_instances = []
        if fr_idx in sample_instances_from.keys():
            # there must be clicks sampled on these instances in this frame
            key_instances = sample_instances_from[fr_idx]
            center_click = first_click_center
        else:
            # optional frames get fg clicks sampled with some probability
            center_click = False

        num_clicks_per_object_fr, fg_coords_list_fr, t, count = get_foreground_clicks(
                                                            fr_idx,                     # sample clicks from this frame
                                                            instance_ids,               # IDs of the instances present in the clip
                                                            instance_masks[fr_idx],     # binary instance masks of the frame
                                                            key_instances,              # mandatorily add clicks on these instances
                                                            optional_frames_fg_prob,    # if there's no key instances, optionally sample from this frame
                                                            center_click,               # whether to add the 1st click at object center or not
                                                            max_num_points,             # max #clicks to sample
                                                            gamma,                      # probability (down)scaling factor for sampling clicks
                                                            t,                          # current timestamp
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
            bg_prob += bg_prob * gamma
            bg_coords_list.append([])
            continue

        bg_coords_list_fr, t = get_background_clicks(
                                            fr_idx,
                                            bg_masks[fr_idx],
                                            max_num_points+1,
                                            gamma,
                                            t
                                        )
        bg_coords_list.append(bg_coords_list_fr)
        if len(bg_coords_list_fr) > 0:
            max_timestamp_clip[fr_idx]=t-1


    return num_clicks_per_object.tolist(), fg_coords_list, bg_coords_list, max_timestamp_clip


def get_clicks_coords_evaluation(
    instance_masks,
    clip_instance_ids,
    sequence_instance_ids,
    frame_instance_occupancy,
    max_num_points=1,
    first_click_center=True,
    start_t=1
):
    """
    Clicker for evaluation data

    Args:
        instance_ids: IDs of instances present in each frame
        instance_masks: binary instance masks of the frames
        max_num_points: maximum number of points to sample *for each instance*
        first_click_center: whether to sample first click at object center, (default: True)
    """
    num_frames = instance_masks.shape[0]

    fg_coords_list = [[] for _ in range(num_frames)]
    bg_coords_list = [[] for _ in range(num_frames)]
    max_timestamp = [0 for _ in range(num_frames)]

    # sample one click (object center) for each instance
    # across all frames
    sample_instances_from = defaultdict(list)
    for inst_id, frame_idxs in frame_instance_occupancy.items():
        choice = random.choice(frame_idxs)
        # add a click on instance `inst_id` in frame `choice`
        sample_instances_from[choice].append(inst_id)

    
    # all clicks in a clip share a single timeline
    t = start_t
    num_clicks_per_object = np.zeros((num_frames, len(sequence_instance_ids))).astype('int')
    # Sample clicks from a given frame
    for fr_idx in range(num_frames):

        if fr_idx not in sample_instances_from.keys():
            continue
        
        # instance masks of the frame
        fr_mask = instance_masks[fr_idx]
        # sample one click at object center
        for inst_id in sample_instances_from[fr_idx]:
            _mask = fr_mask[inst_id-1]
            # fetch center coordinates
            center_coords = get_center_coords(_mask)
            # record click
            fg_coords_list[fr_idx].append([center_coords[0], center_coords[1], inst_id, fr_idx, t])
            
            num_clicks_per_object[fr_idx][inst_id-1]+=1
            max_timestamp[fr_idx] = t
            t+=1

    return num_clicks_per_object.tolist(), fg_coords_list, bg_coords_list, max_timestamp