import numpy as np
import random
import time
import torch
import torch.nn as nn

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from tqdm import tqdm
from typing import List, Mapping, Optional, Tuple

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.evaluation.manager import SequenceManager
from dynamite_video.evaluation.metrics.metrics import compute_stq
from dynamite_video.model.predictor import Predictor


def evaluate(model: nn.Module, 
             dataset: List[GenericVideoSequence],
             dataset_meta: Mapping,
             tfms: Mapping,
             iou_threshold: float,
             max_interactions: int,
             max_rounds: int,
             output_dir: str,
             seed_id: int=0,
             save_vis: bool=False,
             ):
    """
    Evaluate model on provided dataset.

    Args:
        model: model to evaluate
        dataset: pre-processed dataset is a list where each entry corresponds to one
                sequence in the dataset. Each sequence has been decomposed into a series
                of (potentially overlapping) clips
        iou_threshold: desired IoU value for each target mask (float, default: 0.85)
        max_interactions: max #interactions permitted per target (int)
        max_rounds: max #corrective rounds (int, default: 3)
        seed_id: fixed seed during evaluation (int, default 0)
        output_path: path to store frame predictions across iterations
        save_vis: whether to save visualization of masks with corrective clicks. Visualizations
                are saved in `output_path` (bool, default False)
    """
    
    logger = setup_logger(output=output_dir, distributed_rank=comm.get_rank(), name="Interactive Evaluation")
    logger.info(f"Starting inference on {len(dataset)} sequences...")
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        random.seed(123456+seed_id)

        dataset_scores = defaultdict(dict)
        
        for i, video in enumerate(dataset):
            logger.info(f"Processing Sequence {video.id} [{i+1}/{len(dataset)}]")
            
            # sequence manager for current sequence
            manager = SequenceManager(video, dataset_meta, tfms, output_dir, save_vis)

            # a fresh model for each sequence
            predictor = Predictor(model)

            # click budget for whole sequence = max #clicks per target * #targets
            click_budget = max_interactions * manager.N

            # time-keeping in each round
            propagation_time = 0
            metric_compute_time = 0

            # first round starts at first frame
            lowest_frame_index = 0
            while lowest_frame_index!=-1:
                
                manager.round_num += 1
                logger.info(f"Round {manager.round_num}: \nPropagating...")

                # generate indices of shorter sub-sequences or clips from the whole sequence
                clip_indices = manager.generate_clip_indices(start=0) #lowest_frame_index)
                
                ### PROPAGATION ###
                propagation_start_time = time.perf_counter()
                # for clip_idx, indices in enumerate(tqdm(clip_indices, leave=False, desc="Clip")):
                for clip_idx, indices in enumerate(clip_indices):

                    # prepare clip input to the model
                    inputs = manager.extract_clip(indices, clip_idx)
                    # model forward pass
                    binary_pred_masks, pred_logits = predictor.get_prediction([inputs])    # T,N,H,W
                    # store as panoptic prediction
                    manager.store_prediction(binary_pred_masks, pred_logits, clip_idx)

                propagation_time += time.perf_counter() - propagation_start_time

                
                ### EVALUATION METRICS ### 
                logger.info(f"Calculating evaluation metrics...")
                scores, target_level_scores, compute_time = calculate_score(manager)
                metric_compute_time += compute_time
                logger.info(f"STQ scores: {scores}")

                
                ### REFINEMENT ###
                if manager.round_num == max_rounds:
                    logger.info(f'Maximum round limit ({max_rounds}) reached!')
                    lowest_frame_index = -1
                
                elif click_budget <= manager.num_clicks_per_frame.sum():
                    logger.info(f'Click budget ({max_interactions} per frame) over!')
                    lowest_frame_index = -1
                
                else:
                    # select the target with weakest mIoU
                    refine_target, refine_frame = find_refinement_target(target_level_scores, refine_object_selection="worst")

                    # here's another choice one must make. Should the threshold be on a frame-level or on an object level
                    # If all objects meet threshold, then all frames must as well
                    # Even if all frames meet threshold, some objects might not meet IoU threshold
                    # Former is more stringent. And if we have click budget to spare, then why not?
                    # this choice has direct implication on `interaction_metrics()`
                    if refine_target[1] >= iou_threshold:
                        logger.info(f'All targets meet IoU threshold {iou_threshold}!')
                        lowest_frame_index = -1
                    else:
                        # sample a corrective click
                        refined_tgt_id = manager.get_corrective_click(frame_idx=refine_frame[0], refine_tgt_id=refine_target[0])
                        lowest_frame_index = refine_frame[0]
                        logger.info(f'Sampled a click on target {refined_tgt_id} in frame {lowest_frame_index}')

            # interaction metrics
            click_scores = interaction_metrics(manager, target_level_scores, iou_threshold)
            logger.info(f"Click scores: {click_scores}")
            dataset_scores[manager.sequence.id] = scores | click_scores
            
            logger.info(f"{manager.sequence.id}, Time analysis: \
                        \nAverage propagation time per round: {propagation_time/manager.round_num} \
                        \nAverage metric computation time per round: {metric_compute_time/manager.round_num}")
            del manager, predictor
        
        return dataset_scores
    

def calculate_score(manager: SequenceManager) -> Tuple[Mapping, float]:
    start_time = time.perf_counter()
    result = compute_stq(y_true=manager.gt_masks, 
                        y_pred=manager.pred_masks, 
                        target_ids=manager.target_ids,
                        ignore_label=manager.bg_id)
    end_time = time.perf_counter()
    
    scores = {
        "Round": manager.round_num, 
        "#frames": manager.T, 
        "#targets": manager.N, 
        "#clicks": int(manager.num_clicks_per_frame.sum()), 
        "STQ": result["STQ"], 
        "AQ": result["AQ"], 
        "SQ": result["SQ"],
    }
    target_level_scores = {
        "sq_per_target": result["sq_per_target"],
        "sq_per_frame_per_target": result["sq_per_frame_per_target"]
    }
    return scores, target_level_scores, end_time - start_time


def find_refinement_target(
        target_level_scores: Mapping, 
        refine_object_selection: str="worst", 
        K: Optional[int]=None, 
        iou_threshold: Optional[float]=None,
):
    """
    The worst target object can be determined based on it's low score on AQ or SQ metric. 
    Which one to choose? AQ and SQ have high positive correlation, so either choice seems fine. 
    Empirically, it can be been seen that in some cases, the object with half-decent AQ score 
    has very poor SQ. This may indicate that something is tracked, even if the masks are noisy.
    Moreover, a low AQ but high SQ case - which is rare - is arguably less bad than the reverse,
    where at least the object is correctly segmented, even if tracking is flaky. So, we choose SQ.

    Args:
        * target_level_scores: dict with "sq_per_target", "sq_per_frame"
        * refine_object_selection: strategy to select target object(s) to refine. Choose from: 
            "worst": select the single worst target object
            "topk": select top-K worst target objects
            "threshold": select all target objects with SQ lower than a threshold
        * K: int, value of K when `refine_object_selection` is "topk"
        * iou_threshold: float, IoU threshold when `refine_object_selection` is "threshold"
    """
    sq_per_target = target_level_scores["sq_per_target"]                                    # N
    sq_per_frame_per_target = np.asarray(target_level_scores["sq_per_frame_per_target"])    # T,N

    if refine_object_selection=="worst":
        # minimum sq score obtained by a target
        min_tgt_sq = np.min(sq_per_target)
        # the target object with the worst sq score
        worst_target = np.argmin(sq_per_target)
        refine_target = [worst_target + 1, min_tgt_sq]

        # frame where this target has lowest IoU
        min_fr_sq = np.min(sq_per_frame_per_target[:, worst_target])
        worst_frame = np.argmin(sq_per_frame_per_target[:, worst_target])
        refine_frame = [worst_frame, min_fr_sq]
    
    elif refine_object_selection=="topk":
        raise NotImplementedError
    elif refine_object_selection=="threshold":
        raise NotImplementedError
    elif refine_object_selection=="first_drop":
        raise NotImplementedError
    else:
        raise RuntimeError(f"refine_object_selection strategy must be one of ['worst', 'topk', 'threshold'], got {refine_object_selection}")

    return refine_target, refine_frame


def interaction_metrics(manager: SequenceManager, target_level_scores: Mapping, iou_threshold: float):
    """
    Compute interaction cost

    1. PFO - % failed objects, objects that did not reach the IoU threshold in the whole video
    2. PMO - % missing objects, objects that were completely missed across the whole video
    3. PFF - % failed frames, frames that did not reach the IoU threshold
    4. NCI - #clicks per image, normalized by #target objects in it
        Just computing average #clicks (NoC) per object or per frame undermines
        the multi-instance setup or the temporal propagation
    """
    sq_per_target = target_level_scores["sq_per_target"]
    sq_per_frame_per_target = target_level_scores["sq_per_frame_per_target"]

    # num of target objects with IoU lower than threshold
    PFO = len(np.where(sq_per_target < iou_threshold)[0]) / manager.N
    # num of target objects with IoU == 0
    PMO = len(np.where(sq_per_target == 0.)[0]) / manager.N
    
    PFF = 0
    NCI = 0.
    
    for fr_idx, fr_scores in enumerate(sq_per_frame_per_target):
        
        # num of clicks per frame, normalized by the num of targets
        num_targets = np.count_nonzero(fr_scores!=2.)
        NCI += float(manager.num_clicks_per_frame[fr_idx]) / num_targets
        
        filtered_scores = fr_scores[fr_scores != 2.0]
        if np.mean(filtered_scores) < iou_threshold:
            PFF += 1

    NCI/=manager.T
    PFF/=manager.T

    return {"PFO": round(PFO, 2), "PMO": round(PMO, 2), "PFF": round(PFF, 2), "NCI": round(NCI, 2)}


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)