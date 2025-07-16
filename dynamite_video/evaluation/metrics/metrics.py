import numpy as np
import tensorflow as tf

from typing import Dict, List, Tuple

from dynamite_video.evaluation.metrics import batched_f_measure, batched_jaccard, STQuality


def compute_stq(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        object_ids: List,
        ignore_label: int,
        mapping,        # debug
        dataset_meta    # debug
) -> Tuple:
    """
    Compute STQ for current sequence

    Args:
        y_true: ground truth labels, np.ndarray, shape T,H,W
        y_pred: predicted labels, np.ndarray, shape T,H,W
        dataset_meta: dataset-specific dictionary with following keys:
            "num_classes": number of semantic classes, e.g., 19 in KITTI-STEP
            "things_list": list of 'thing' classes
            "ignore_label": int specifying ignore class (VOID) label
            "max_instances_per_category": max num instances per semantic class
    """
    # convert serially labeled masks to the semantic_class * 1000 + instance_id scheme
    T,H,W = y_true.shape
    
    ignore_label = dataset_meta["num_classes"] # 255000
    remapped_y_true = np.full((T,H,W), fill_value=ignore_label, dtype=np.uint32)
    for serial_id, orig_id in mapping.items():
        remapped_y_true[np.where(y_true==serial_id)] = serial_id - 1
    
    remapped_y_pred = np.full((T,H,W), fill_value=ignore_label, dtype=np.uint32)
    for serial_id, orig_id in mapping.items():
        remapped_y_pred[np.where(y_true==serial_id)] = serial_id - 1

    
    stq_metric = STQuality(num_classes=len(object_ids), #dataset_meta["num_classes"],
                           things_list=object_ids, #dataset_meta["things_list"],
                           ignore_label=ignore_label, #dataset_meta["ignore_class"],
                           max_instances_per_category=1, #dataset_meta["max_instances_per_category"],
                           offset=int(1e6))

    for gt_mask, pred_mask in zip(remapped_y_true, remapped_y_pred):

        # convert to TF tensors
        # TODO: redundant?
        gt_mask = tf.convert_to_tensor(gt_mask, tf.int64)
        pred_mask = tf.convert_to_tensor(pred_mask, tf.int64)

        stq_metric.update_state(gt_mask, pred_mask)

    result = stq_metric.result()

    def np_to_native_type(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif hasattr(x, "item"):
            return x.item()
        else:
            return x

    return (np_to_native_type(result["STQ"]), 
            np_to_native_type(result["AQ"]), 
            np_to_native_type(result["IoU"]))


def compute_j_and_f(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        object_ids: List, 
        ignore_label: int,
) -> Dict:
    """
    Compute J&F score at a frame level

    `object_ids` must not contain VOID class, if applicable for the dataset.

    Args:
        y_true: ground truth labels, np.ndarray, shape T,H,W
        y_pred: predicted labels, np.ndarray, shape T,H,W
        object_ids: list of IDs of objects present in `y_true` 
        ign_label: label of ignore class
    """

    if y_true.ndim != 3:
        raise ValueError('y_true array must have 3 dimensions.')
    if y_pred.ndim != 3:
        raise ValueError('y_pred array must have 3 dimensions.')
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))
    if len(object_ids) == 0:
        raise ValueError('Number of objects in y_true should be higher than 0.')

    jaccard = batched_jaccard(y_true, y_pred, object_ids, ignore_label)
    f_measure = batched_f_measure(y_true, y_pred, object_ids, ignore_label)
    
    jaccard_mean_per_frame = jaccard.mean(axis=1)
    f_measure_mean_per_frame = f_measure.mean(axis=1)
    
    j_and_f = 0.5*jaccard_mean_per_frame + 0.5*f_measure_mean_per_frame

    return {
        "jaccard_mean_per_frame": jaccard_mean_per_frame,
        "jaccard": jaccard,
        "f_measure_mean_per_frame": f_measure_mean_per_frame,
        "f_measure": f_measure,
        "j_and_f": j_and_f
    }