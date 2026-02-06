import numpy as np
import random
import time
import torch
import torch.nn as nn

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
             connected_component_sampling: bool=True,
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

        dataset_scores = []
        dataset_target_scores = []
        
        for i, video in enumerate(dataset):
            logger.info(f"Processing Sequence {video.id} [{i+1}/{len(dataset)}]")
            
            # sequence manager for current sequence
            manager = SequenceManager(video, dataset_meta, tfms, output_dir, save_vis, min_mask_area, connected_component_sampling)
            del video

            # a fresh model for each sequence
            predictor = Predictor(model)

            # click budget for whole sequence = max #clicks per target * #targets
            _ = manager.set_budget(max_interactions)

            # time-keeping in each round
            propagation_time = 0
            metric_compute_time = 0
            lowest_frame_index = 0
            round_scores = []

            # loop until click budget is not over
            while True:

                manager.round_num += 1
                logger.info(f"Round {manager.round_num}: Propagating...")

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
                round_scores.append(target_level_scores["sq_per_target"])

                # Stopping criterion 1: all targets meet threshold
                if min(target_level_scores["sq_per_target"]) >= iou_threshold:
                    logger.info(f'All targets meet IoU threshold!')
                    break

                # Stopping criterion 2: budget for every weak target is over
                remaining_budgets = manager.budget[np.where(target_level_scores["sq_per_target"]<iou_threshold)]
                if remaining_budgets.sum() <= 0:
                    logger.info(f'Click budget is over for all targets with IoU < threshold!')
                    break

                # Stopping criterion 3: last round
                if manager.round_num == max_rounds:
                    break

                ### REFINEMENT ###
                manager.find_refinement_targets(target_level_scores, iou_threshold, eval_strategy)
                
            # interaction metrics
            click_scores, target_scores = interaction_metrics(manager, target_level_scores, iou_threshold, max_interactions)
            logger.info(f"Click scores: {click_scores}")
            dataset_scores.append({"Name": manager.sequence.id} | scores | click_scores)
            dataset_target_scores.append(target_scores)
            
            logger.info(f"{manager.sequence.id}, Time analysis: \
                        \nAverage propagation time per round: {propagation_time/manager.round_num} \
                        \nAverage metric computation time per round: {metric_compute_time/manager.round_num}")
            del manager, predictor
        
        return dataset_scores, dataset_target_scores


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
    
    click_scores = {"PFO": round(PFO, 2), "PMO": round(PMO, 2), "NoC": round(NoC, 2)}
    target_scores = {
            "num_clicks_per_target": manager.num_clicks_per_target.sum(axis=0).tolist(),
            "sq_per_target": target_level_scores["sq_per_target"].tolist()
        }
    return click_scores, target_scores
    
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