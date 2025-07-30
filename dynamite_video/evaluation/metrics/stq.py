# Adapted by Srijeet Roy from "Implementation of the Segmentation and Tracking Quality (STQ) metric" by 
# The Deeplab2 Authors: https://github.com/google-research/deeplab2/blob/main/evaluation/segmentation_and_tracking_quality.py

from typing import Any, Dict, MutableMapping, Optional, Sequence, Text

import warnings

import numpy as np
import tensorflow as tf

def _check_weights(unique_weight_list: Sequence[float]):
  if not set(unique_weight_list).issubset({0.5, 1.0}):
    warnings.warn(
        'Potential performance degration as the code is not optimized'
        ' when weights has too many different elements.'
    )


def _update_dict_stats(
    stat_dict: MutableMapping[int, tf.Tensor],
    id_array: tf.Tensor,
    weights: Optional[tf.Tensor] = None,
):
  """Updates a given dict with corresponding counts."""
  if weights is None:
    unique_weight_list = [1.0]
  else:
    unique_weight_list, _ = tf.unique(weights)
    unique_weight_list = unique_weight_list.numpy().tolist()
  _check_weights(unique_weight_list)
  # Iterate through the unique weight values, and weighted-average the counts.
  # Example usage: lower the weights in the region covered by multiple camera in
  # panoramic video panoptic segmentation (PVPS).
  for weight in unique_weight_list:
    if weights is None:
      ids, _, counts = tf.unique_with_counts(id_array)
    else:
      ids, _, counts = tf.unique_with_counts(
          tf.boolean_mask(id_array, tf.equal(weight, weights)))
    for idx, count in zip(ids.numpy(), tf.cast(counts, tf.float32)):
      if idx in stat_dict:
        stat_dict[idx] += count * weight
      else:
        stat_dict[idx] = count * weight


class STQuality(object):
  """Metric class for the Segmentation and Tracking Quality (STQ).

  The metric computes the geometric mean of two terms.
  - Association Quality: This term measures the quality of the track ID
      assignment for `thing` classes. It is formulated as a weighted IoU
      measure.
  - Segmentation Quality: This term measures the semantic segmentation quality.
      The standard class IoU measure is used for this.
  """

  def __init__(self,
               num_classes: int,
               things_list: Sequence[int],
               ignore_label: int,
               offset: int
               ):
    """Initialization of the STQ metric.

    Args:
      num_classes: Number of classes in the dataset as an integer.
      things_list: A sequence of class ids that belong to `things`.
      ignore_label: The class id to be ignored in evaluation as an integer or
        integer tensor.
      offset: The maximum number of unique labels as an integer or integer
        tensor.
      name: An optional name. (default: 'st_quality')
    """
    self._num_classes = num_classes
    self._ignore_label = ignore_label
    self._things_list = things_list

    self._confusion_matrix_size = num_classes
    self._include_indices = np.array([i for i in range(num_classes) if i != self._ignore_label])

    self._iou_confusion_matrix = None
    self._predictions = None
    self._ground_truth = None
    self._intersections = None
    self._ious = []
    
    self._offset = offset
    if offset < num_classes:
      raise ValueError('The provided offset %d is too small. No guarantess '
                       'about the correctness of the results can be made. '
                       'Please choose an offset that is higher than num_classes'
                       ' = %d' % num_classes)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   weights: Optional[tf.Tensor] = None):
    """Accumulates the segmentation and tracking quality statistics.

    Args:
      y_true: The ground-truth panoptic label map for a particular video frame
      y_pred: The predicted panoptic label map for a particular video frame
      weights: The weights for each pixel with the same shape of `y_true`
    """
    y_true = tf.cast(y_true, dtype=tf.int64)
    y_pred = tf.cast(y_pred, dtype=tf.int64)
    if weights is not None:
      weights = tf.reshape(weights, y_true.shape)
    
    # to compute SQ, accumulate the target mask intersections in a confusion matrix
    
    # semantic labels are the same as the target IDs
    semantic_label = tf.identity(y_true)
    semantic_prediction = tf.identity(y_pred)
    
    # confusion matrix computes the area of intersection of each 
    # target mask in the ground truth and predicted panoptic masks
    cm = tf.math.confusion_matrix(
                labels=tf.reshape(semantic_label, [-1]),
                predictions=tf.reshape(semantic_prediction, [-1]),
                num_classes=self._confusion_matrix_size,
                weights=tf.reshape(weights, [-1]) if weights is not None else None,
                dtype=tf.float64
            )
    
    # record frame-level IoUs for each target
    confusion = cm.numpy()
    intersections = confusion.diagonal()
    fps = confusion.sum(axis=0) - intersections
    fns = confusion.sum(axis=1) - intersections
    unions = intersections + fps + fns
    ious = (intersections.astype(np.double) / np.maximum(unions, 1e-15).astype(np.double))
    ious[np.where(unions==0)] = 2.
    self._ious.append(ious)
    
    # accumulate the confusion matrix scores (area of intersection) across frames
    if self._iou_confusion_matrix is not None:
        self._iou_confusion_matrix += cm
    else:
        self._iou_confusion_matrix = cm
        self._predictions = {}
        self._ground_truth = {}
        self._intersections = {}
    
    # to compute AQ, save the target masks

    # separate thing targets - in this case, everywhere other than area labeled 0
    instance_label = tf.identity(y_true)

    label_mask = tf.zeros_like(semantic_label, dtype=tf.bool)
    prediction_mask = tf.zeros_like(semantic_prediction, dtype=tf.bool)
    for things_class_id in self._things_list:
      label_mask = tf.logical_or(label_mask, tf.equal(semantic_label, things_class_id))
      prediction_mask = tf.logical_or(prediction_mask, tf.equal(semantic_prediction, things_class_id))

    # Select the `crowd` region of the current class. This region is encoded instance id `0`.
    is_crowd = tf.logical_and(tf.equal(instance_label, 0), label_mask)
    # Select the non-crowd region of the corresponding class as the `crowd` region is ignored for the tracking term.
    label_mask = tf.logical_and(label_mask, tf.logical_not(is_crowd))
    # Do not punish id assignment for regions that are annotated as `crowd` in the ground-truth.
    prediction_mask = tf.logical_and(prediction_mask, tf.logical_not(is_crowd))

    seq_preds = self._predictions
    seq_gts = self._ground_truth
    seq_intersects = self._intersections

    # Compute and update areas of ground-truth, predictions and intersections.
    _update_dict_stats(seq_preds, y_pred[prediction_mask], weights[prediction_mask] if weights is not None else None)
    _update_dict_stats(seq_gts, y_true[label_mask],weights[label_mask] if weights is not None else None)
    
    # store the intersection between every g.t. target and predicted target the area of intersection between g.t. target 
    # with ID g and predicted target with ID p, is stored in the dict w key (g * _offset + p) (e.g., if g.t. target ID is 
    # 9 and predicted target ID is 5 then, the corresponding key in the intersection dict is 9000005, where offset = 10^6)
    non_crowd_intersection = tf.logical_and(label_mask, prediction_mask)
    intersection_ids = (y_true[non_crowd_intersection] * self._offset + y_pred[non_crowd_intersection])
    _update_dict_stats(seq_intersects, intersection_ids, weights[non_crowd_intersection] if weights is not None else None)


  def result(self) -> Dict[Text, Any]:
    """Computes the segmentation and tracking quality.

    Returns:
      A dictionary containing:
        - 'STQ': The total STQ score.
        - 'AQ': The total association quality (AQ) score.
        - 'IoU': The total mean IoU.
        - 'refine_target': ID of the target with the lowest AQ.
        - 'refine_frame': index of the frame where `refine_target` has lowest IoU
    """
    # Compute association quality (AQ)
    aq_per_tgt = []
    # get AQ for each gt target
    for gt_id, gt_size in self._ground_truth.items():
        inner_sum = 0.0
        for pr_id, pr_size in self._predictions.items():
            tpa_key = self._offset * gt_id + pr_id
            if tpa_key in self._intersections:
                # how much gt target intersected with certain pred target
                tpa = self._intersections[tpa_key].numpy()
                # how much the prediction overflowed
                fpa = pr_size.numpy() - tpa
                # how much of the gt was missed
                fna = gt_size.numpy() - tpa
                inner_sum += tpa * (tpa / (tpa + fpa + fna))
        aq_per_tgt.append(1.0 / gt_size.numpy() * inner_sum)

    aq_mean = sum(aq_per_tgt) / max(len(self._ground_truth), 1e-15)
    # minimum aq score obtained by a target
    min_aq = min(aq_per_tgt)
    # print(f"AQ per target: {aq_per_tgt}")
    # the target with the minimum aq score
    refine_target = list(self._ground_truth.keys())[aq_per_tgt.index(min_aq)]
    # print(f"Minimum AQ: {min_aq} for target {refine_target}")

    # Compute IoU scores.
    # The rows correspond to ground-truth and the columns to predictions.
    # Remove fp from confusion matrix for the void/ignore class.
    confusion = self._iou_confusion_matrix.numpy()
    removal_matrix = np.zeros_like(confusion)
    removal_matrix[self._include_indices, :] = 1.0
    confusion *= removal_matrix

    # `intersections` corresponds to true positives.
    intersections = confusion.diagonal()
    fps = confusion.sum(axis=0) - intersections
    fns = confusion.sum(axis=1) - intersections
    unions = intersections + fps + fns

    num_classes = np.count_nonzero(unions)
    ious = (intersections.astype(np.double) /
            np.maximum(unions, 1e-15).astype(np.double))
    iou_mean = np.sum(ious) / num_classes

    # frame where this target has lowest IoU
    frame_level_ious = np.asarray(self._ious)[:, refine_target]
    # print(f"Frame level IoUs for target {refine_target}: {frame_level_ious}")
    min_iou = np.min(frame_level_ious)
    refine_frame = np.argmin(frame_level_ious)
    # print(f"Minimum IoU: {min_iou} at frame {refine_frame}")

    st_quality = np.sqrt(aq_mean * iou_mean)
    return {'STQ': st_quality,
            'AQ': aq_mean,
            'IoU': float(iou_mean),
            "refine_target": [refine_target, min_aq],
            "refine_frame": [refine_frame, min_iou]
            }
