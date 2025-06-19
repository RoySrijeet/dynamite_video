import numpy as np
import tensorflow as tf

from dynamite_video.evaluation.metrics import batched_f_measure, batched_jaccard, STQuality


def compute_stq(gt_masks, pred_masks, dataset_meta):
    """
    Compute STQ for current sequence

    Args:
        gt_masks: T,H,W
        pred_masks: T,H,W
        dataset_meta: dataset-specific dictionary with following keys:
            "num_classes": number of semantic classes, e.g., 19 in KITTI-STEP
            "things_list": list of 'thing' classes
            "ignore_label": int specifying ignore class (VOID) label
            "max_instances_per_category": max num instances per semantic class
    """
    stq_metric = STQuality(num_classes=dataset_meta["num_classes"],
                           things_list=dataset_meta["things_list"],
                           ignore_label=dataset_meta["ignore_class"],
                           max_instances_per_category=dataset_meta["max_instances_per_category"],
                           offset=int(1e6))

    for frame_gt, frame_pred in zip(gt_masks, pred_masks):

        # STQ expects pixel labels to have the following format:
        # semantic_map * max_instances_per_category + instance_map

        # convert to TF tensors
        frame_pred = tf.convert_to_tensor(frame_pred, tf.int64)
        frame_gt = tf.convert_to_tensor(frame_gt, tf.int64)

        stq_metric.update_state(frame_gt, frame_pred)

    result = stq_metric.result()
    for k, v in result.items():
        print(f"{k}: {v}")

    def np_to_native_type(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif hasattr(x, "item"):
            return x.item()
        else:
            return x

    result = {k: np_to_native_type(v) for k, v in result.items()}
    result["IoU_per_seq"] = [x.item() for x in result["IoU_per_seq"]]

    return result
