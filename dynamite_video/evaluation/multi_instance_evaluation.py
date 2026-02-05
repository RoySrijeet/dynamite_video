import numpy as np
import random
import time
import torch
import torch.nn as nn

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from tqdm import tqdm
from typing import List, Mapping

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.data.generic_video_parser import GenericVideoSequence
from dynamite_video.evaluation.manager import SequenceManager
from dynamite_video.model.predictor import Predictor

def evaluate(model: nn.Module, 
             dataset: List[GenericVideoSequence],
             dataset_meta: Mapping,
             tfms: Mapping,
             iou_threshold: float=0.85,
             max_interactions: int=10,
             max_rounds: int=3,
             eval_strategy:str="worst",
             min_mask_area: int=200,
             output_dir: str="",
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
        dataset_click_scores = []
        
        for i, video in enumerate(dataset):
            logger.info(f"Processing Sequence {video.id} [{i+1}/{len(dataset)}]")
            
            # sequence manager for current sequence
            manager = SequenceManager(video, dataset_meta, tfms, output_dir, save_vis, min_mask_area)
            del video

            # a fresh model for each sequence
            predictor = Predictor(model)

            # click budget for whole sequence = max #clicks per target * #targets
            click_budget = max_interactions * manager.N

            # time-keeping in each round
            propagation_time = 0
            metric_compute_time = 0
            lowest_frame_index = 0

            # loop until click budget is not over
            for _ in range(max_rounds): #while True:

                manager.round_num += 1
                logger.info(f"Round {manager.round_num}: \nPropagating...")

                # generate indices of shorter sub-sequences or clips from the whole sequence
                clip_indices = manager.generate_clip_indices(start=lowest_frame_index)
                
                ### PROPAGATION ###
                propagation_start_time = time.perf_counter()
                for clip_idx, indices in enumerate(tqdm(clip_indices, leave=False, desc="Clip")):

                    # prepare clip input to the model
                    inputs = manager.extract_clip(indices)
                    # model forward pass
                    binary_pred_masks, pred_logits = predictor.get_prediction([inputs])    # T,N,H,W
                    # store as panoptic prediction
                    manager.store_prediction(binary_pred_masks, pred_logits, clip_idx)

                propagation_time += time.perf_counter() - propagation_start_time
                
                ### EVALUATION METRICS ### 
                logger.info(f"Calculating evaluation metrics...")
                start_time = time.perf_counter()
                scores, target_level_scores = manager.calculate_score()
                metric_compute_time += time.perf_counter() - start_time
                logger.info(f"STQ scores: {scores}")

                # Stopping criterion 1: all targets meet threshold
                if min(target_level_scores["sq_per_target"]) >= iou_threshold:
                    logger.info(f'All targets meet IoU threshold!')
                    break

                # Stopping criterion 2: budget is over
                remaining_click_budget = click_budget - scores["#clicks"]
                if remaining_click_budget <= 0:
                    logger.info(f'Click budget is over!')
                    break

                ### REFINEMENT ###
                # find refinement targets - one corrective click on each target object with mIoU < threshold
                refinements = find_refinement_targets(
                        target_level_scores,
                        budget = remaining_click_budget,
                        iou_threshold=iou_threshold,
                        eval_strategy=eval_strategy
                    )
                
                for refine_target, refine_frame in refinements:
                    refined_tgt_id = manager.get_corrective_click(frame_idx=refine_frame, refine_tgt_id=refine_target)
                    logger.info(f'Sampled a click on target {refined_tgt_id} in frame {refine_frame} to refine existing mask of target {refine_target}')
                
            # interaction metrics
            click_scores, ds_level_entry = interaction_metrics(manager, target_level_scores, iou_threshold)
            logger.info(f"Click scores: {click_scores}")
            dataset_scores[manager.sequence.id] = scores | click_scores
            dataset_click_scores.append(ds_level_entry)
            
            logger.info(f"{manager.sequence.id}, Time analysis: \
                        \nAverage propagation time per round: {propagation_time/manager.round_num} \
                        \nAverage metric computation time per round: {metric_compute_time/manager.round_num}")
            del manager, predictor
        
        return dataset_scores


def find_refinement_targets(
        target_level_scores: Mapping, 
        budget: int, 
        iou_threshold: float,
        eval_strategy: str,
):
    """
    Add one corrective click on each target with SQ (mIoU) < IoU threshold. If there's not 
    enough budget left, add click on as many objects as possible with priority given to the 
    objects with larger error.
    
    Selection method: 
    * selecting the object: all objects with mIoU less than threshold are candidates

    * selecting the frame: with lowest SQ for each target to be refined

    If budget does not allow one click per object to be refined, select the worst ones.

    Args:
        * target_level_scores: object level SQ scores
        * budget: int, available click budget
        * iou_threshold: float, IoU threshold
        * eval_strategy: "worst" to select only the worst candidate, 
                        "all" to select all candidates under `iou_threshold`
    """
    # mIoU of target objects across the entire sequence
    sq_per_target = target_level_scores["sq_per_target"]    # N
    # IoU of target objects in each frame
    sq_per_frame_per_target = np.asarray(target_level_scores["sq_per_frame_per_target"])    # T,N
    # error_per_frame_per_target = np.asarray(target_level_scores["error_per_frame_per_target"])

    if eval_strategy == "worst":
        #candidate_objects = [np.argmin(sq_per_target)]
        all_candidate_objects = np.where(sq_per_target < iou_threshold)[0]
        candidate_objects = [np.random.choice(all_candidate_objects)]
    else:
        # target objects with mIoU lower than threshold
        all_candidate_objects = np.where(sq_per_target < iou_threshold)[0]
        if budget < len(all_candidate_objects):
            # if there's not enough budget, pick the weakest objects to exhaust the budget
            candidate_objects = np.argpartition(sq_per_target, budget)
        else:
            candidate_objects = all_candidate_objects

    # bias toward choosing earlier frames
    alpha = 0.5 * (1 - iou_threshold)

    refinements = []
    for obj_id in candidate_objects:
        
        # frames where the object has IoU < threshold
        candidate_frames = sq_per_frame_per_target[:, obj_id] < iou_threshold

        # frame intervals
        diff = np.diff(candidate_frames.astype(int))
        starts = np.where(diff == 1)[0] + 1
        if candidate_frames[0]:
            starts = np.r_[0, starts]
        ends = np.where(diff == -1)[0]
        if candidate_frames[-1]:
            ends = np.r_[ends, len(candidate_frames) - 1]
        weak_intervals = list(zip(starts, ends))

        if len(weak_intervals) == 0:
            raise RuntimeError(f"No weak frame intervals found for an object to be refined!")
        
        # severity scores
        best_score = -float("inf")
        best_start = None

        for start, end in weak_intervals:
            # how long the interval is
            length = end - start + 1
            # how bad the masks in the interval are
            mean_iou = sq_per_frame_per_target[start:end+1, obj_id].mean()
            # how harmful the interval is
            severity = length * (1.0 - mean_iou)
            # penalize later intervals
            score = severity - alpha * start

            if score > best_score:
                best_score = score
                best_start = start

        refinements.append([obj_id+1, best_start])

    return refinements


def interaction_metrics(
        manager: SequenceManager, 
        target_level_scores: Mapping, 
        iou_threshold: float,
        max_interactions: int,
):
    """
    Compute interaction cost

    1. NoC - average #clicks required for each object to reach IoU threshold. If threshold is 
             not reached, it is set to the click budget (`max_interactions`)
    2. PFO - % failed objects, objects that did not reach the IoU threshold
    3. PMO - % missing objects, objects that were completely missed across the whole video
    """
    sq_per_target = target_level_scores["sq_per_target"]
    sq_per_frame_per_target = target_level_scores["sq_per_frame_per_target"]

    # num of target objects with IoU lower than threshold
    PFO = len(np.where(sq_per_target < iou_threshold)[0]) / manager.N
    # num of target objects with IoU == 0
    PMO = len(np.where(sq_per_target == 0.)[0]) / manager.N

    NoC = 0
    for obj_id, obj_sq in enumerate(sq_per_target):
        num_clicks = manager.num_clicks_per_target[:, obj_id].sum()
        if obj_sq >= iou_threshold:
            NoC += num_clicks
        else:
            NoC += max_interactions
    NoC = NoC/manager.N
    
    seq_level_scores = {"PFO": round(PFO, 2), "PMO": round(PMO, 2), "NoC": round(NoC, 2)}
    ds_level_entry = {
            "sequence": manager.sequence.id,
            "T": manager.T,
            "N": manager.N,
            "num_clicks_per_target": manager.num_clicks_per_target,
            "sq_per_target": target_level_scores["sq_per_target"],
            "sq_per_frame_per_target": target_level_scores["sq_per_frame_per_target"],
        }
    return seq_level_scores, ds_level_entry
    
    # PFF - % failed frames, frames that did not reach the IoU threshold
    # NCI - #clicks per image, normalized by #target objects in it

    # PFF = 0
    # NCI = 0.
    
    # for fr_idx, fr_scores in enumerate(sq_per_frame_per_target):
        
    #     # num of clicks per frame, normalized by the num of targets
    #     num_targets = np.count_nonzero(fr_scores!=2.)
    #     NCI += float(manager.num_clicks_per_target[fr_idx].sum()) / num_targets
        
    #     filtered_scores = fr_scores[fr_scores != 2.0]
    #     if np.mean(filtered_scores) < iou_threshold:
    #         PFF += 1

    # NCI/=manager.T
    # PFF/=manager.T
    #return {"PFO": round(PFO, 2), "PMO": round(PMO, 2), "PFF": round(PFF, 2), "NCI": round(NCI, 2)}



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