import cv2
import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict
from PIL import Image

def create_circular_mask(h, w, centers, radius):
    """
    create a circular mask of radius `radius` about the coordinates
    specified by `centers`
    """

    assert centers is not None
    assert radius is not None

    mask=np.zeros((h,w), dtype=bool) 
    for center in centers:

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)

        mask = mask | (dist_from_center <= radius)
    return mask.astype(np.uint8)


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
    
    # object center
    max_dist = np.max(dt)
    coords_y, coords_x = np.where(dt == max_dist)

    t = len(coords_y) // 2

    return [coords_y[t],coords_x[t]]

def get_component_center_coords(mask, budget, cc=True, gap=1, min_area=200, connectivity=8):
    assert mask.ndim==2
    if torch.is_tensor(mask):
        mask = mask.numpy()
    mask = mask.astype(np.uint8)

    padded_mask = np.pad(mask, ((1, 1), (1, 1)), 'constant')
    dt_mask = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    # use 3 (fast approximate), instead of 0 (exact L2 dist)

    if cc:
        # merge tiny gaps
        k = 2 * gap + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        components = list(range(1, num_labels))
        if budget is not None and budget < len(components):
            components = sorted(components, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
            components = components[:budget]
        centers = []
        for lab in components:
            if stats[lab, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.where(labels == lab)
            if ys.size == 0:
                continue
            # pick the pixel inside this component that is farthest from background
            i = np.argmax(dt_mask[ys, xs])
            centers.append([int(ys[i]), int(xs[i])])
            
        centers = np.array(centers, dtype=float) if centers else np.zeros((0,2), float)    
        if len(centers) > 0:
            return centers.astype(np.int_).tolist()
    
    # object center
    max_dist = np.max(dt_mask)
    coords_y, coords_x = np.where(dt_mask == max_dist)
    t = len(coords_y) // 2
    return [[coords_y[t],coords_x[t]]]


def get_random_coords(mask, n=1):
    """
    Sample clicks randomly from binary mask

    Args:
        mask: binary mask [H, W], np.ndarray
    """
    assert mask.ndim==2

    if torch.is_tensor(mask):
        mask = mask.numpy()
    mask = mask.astype(np.uint8)

    kernel = np.ones((3,3),np.uint8)
    _eroded_m = cv2.erode(mask.copy(), kernel, iterations=1)
    sample_locations = np.argwhere(_eroded_m)
    
    # randomly select clicks
    index = random.sample(range(sample_locations.shape[0]), n)
    clicks = []
    for idx in index:
        extra_coords = sample_locations[idx]
        clicks.append([extra_coords[0], extra_coords[1]])
    return clicks


def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))
color_map = get_palette(200).flatten().tolist()


def save_color_palette(category_labels, path_to_visualization):
    color_map = get_palette(200)
    nrows = len(category_labels)
    fig, ax = plt.subplots(figsize=(3, nrows*0.5))
    for i, (tgt_id, tgt_cls) in enumerate(category_labels.items()):
        color = color_map[tgt_id]
        ax.add_patch(plt.Rectangle((0, nrows - i - 1), 1, 1, color=color / 255))
        ax.text(1.5, nrows - i - 0.5, f"{tgt_id}: " + tgt_cls, ha='left', va='center', fontsize=12)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, nrows)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_visualization, "color_map.png"))
    plt.close()


davis_palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]


def show_points(image, coords, label, marker_size=15):
    if label==0:
        color = (0,0,255)     # red bg
    elif label==1:
        color = (0,255,0)     # green fg
    elif label==2:
        color = (255,255,0)   # cyan for overlapping/correcting clicks

    for c in coords:
        x = c[1]
        y = c[0]
        cv2.drawMarker(
            image,
            position=(int(x), int(y)),
            color=color,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=marker_size,
            thickness=2,
            line_type=cv2.LINE_AA
        )

def serialize_target_ids(orig_ids):
    """
    Serialize target IDs. IDs are 1-indexed to avoid conflict in semantic mask
    with background pixels (0)

    Args:
        orig_ids: original target IDs, potentially non-sequential

    Returns:
        orig_to_serial_id: mapping from original IDs to sequential IDs
        serial_to_orig_id: mapping from sequential IDs to original IDs
    """
    
    serial_ids = [i for i in range(1, len(orig_ids)+1)]
    serial_to_orig_id = OrderedDict(zip(serial_ids, orig_ids))
    orig_to_serial_id = OrderedDict(zip(orig_ids, serial_ids))
    return orig_to_serial_id, serial_to_orig_id


def get_category_labels(orig_to_serial_ids, dataset_meta):
    category_labels = {}
    for orig_id, serial_id in orig_to_serial_ids.items():
        if dataset_meta["dataset"] == "VIPSEG":
            if orig_id in dataset_meta["stuff_list"]:
                tgt_cls = orig_id
                inst_id = -1
            else:
                tgt_cls = orig_id // dataset_meta["max_instances_per_category"]
                inst_id = orig_id % dataset_meta["max_instances_per_category"]
            label = dataset_meta["category_labels"][tgt_cls]
            if inst_id != -1:
                label = label + "_" + str(inst_id)
        else:
            tgt_cls = orig_id // dataset_meta["max_instances_per_category"]
            label = dataset_meta["category_labels"][tgt_cls]
            inst_id = orig_id % dataset_meta["max_instances_per_category"]
            if inst_id != 0:
                label = label + "_" + str(inst_id)
        category_labels[serial_id] = label
    
    return category_labels


# def save_mask_to_disc(masks, binary, mask_names, save_dir):
#     for msk, fname in zip(masks, mask_names):
#         if not binary:
#             msk = Image.fromarray(msk.astype(np.uint8))
#             msk.putpalette(color_map)
#         else:
#             msk = Image.fromarray(msk.astype(np.uint8) * 255)
#         msk.save(os.path.join(save_dir, fname + ".png"))