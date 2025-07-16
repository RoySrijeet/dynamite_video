import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from tqdm import tqdm

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.evaluation.manager import SequenceManager
from dynamite_video.evaluation.metrics.metrics import compute_j_and_f, compute_stq
from dynamite_video.evaluation.predictor import Predictor


def evaluate(cfg, 
             model, 
             dataset,
             dataset_meta,
             iou_threshold=0.85,
             max_interactions=3,
             max_rounds=3,
             eval_strategy="random",
             seed_id=0,
             save_vis=False,
             ):
    """
    Evaluate model on provided dataset.

    Args:
        model: model to evaluate
        dataset: pre-processed dataset is a list where each entry corresponds to one
                sequence in the dataset. Each sequence has been decomposed into a series
                of (potentially overlapping) clips
        iou_threshold: desired IoU value for each object mask (float, default: 0.85)
        max_interactions: max #interactions permitted per object (int, default: 3)
        max_rounds: max #corrective rounds (int, default: 3)
        eval_strategy: strategy to select the instance to add corrective clicks on 
                "worst": select the instance with worst IoU
                "random": select an instance randomly (as long as IoU < iou_threshold)
                "best": select the instance with best IoU
        seed_id: fixed seed during evaluation (int, default 0)
        output_path: path to store frame predictions across iterations
        save_vis: whether to save visualization of masks with corrective clicks. Visualizations
                are saved in `output_path` (bool, default False)
    """
    vis_path = None
    if save_vis:
        vis_path = os.path.join(cfg.OUTPUT_DIR, "vis")
        os.makedirs(vis_path, exist_ok=True)
    dataset_meta["vis_path"] = vis_path

    # debug
    debug_vis_path = os.path.join(cfg.OUTPUT_DIR, "eval_debug")
    os.makedirs(debug_vis_path, exist_ok=True)
    
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="Multi Instance Evaluation")
    logger.info(f"Starting inference on {len(dataset)} sequences...")

    num_static_bg_queries = 0
    if cfg.ITERATIVE.TRAIN.USE_STATIC_BG_QUERIES:
        num_static_bg_queries = cfg.ITERATIVE.TRAIN.NUM_STATIC_BG_QUERIES
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        random.seed(123456+seed_id)

        dataset_stq = defaultdict(list)
        
        for i, video in enumerate(dataset):

            logger.info(f"\nSequence {video.id} [{i+1}/{len(dataset)}]...")

            # a fresh model for each sequence
            predictor = Predictor(model, dataset_meta["num_overlapping_frames"], num_static_bg_queries)
            
            # sequence manager for current sequence
            manager = SequenceManager(video, dataset_meta, cfg.INPUT)

            # click budget for whole sequence
            click_budget = max_interactions * manager.N

            # rounding starts at first frame
            lowest_frame_index = 0

            propagation_time = []
            metric_compute_time = []
            
            while lowest_frame_index!=-1:
                manager.round_num += 1
                logger.info(f"\nRound {manager.round_num}:")

                prop_time = 0

                # generate indices of shorter sub-sequences or clips from the whole sequence
                clip_indices = manager.generate_clip_indices(start=lowest_frame_index)

                ## PROPAGATION ##
                logger.info(f"Predicting {manager.N} masklets in {manager.T} frames...")

                for indices in tqdm(clip_indices, leave=False, desc="Clip"):
                    propagation_start_time = time.perf_counter()
                    
                    inputs = manager.extract_clip(indices)

                    binary_pred_masks, query_init = predictor.get_prediction([inputs], indices)    # T,N,H,W
                    
                    propagation_end_time = time.perf_counter()
                    panoptic_pred_masks = manager.store_prediction(binary_pred_masks, query_init)
                    
                    prop_time+= (propagation_end_time - propagation_start_time)
                propagation_time.append(prop_time)
                
                # metrics
                logger.info(f"Computing metrics...")
                metric_compute_start_time = time.perf_counter()
                video_stq, video_aq, video_sq = compute_stq(y_true=manager.gt_masks, 
                                                             y_pred=manager.pred_masks, 
                                                             object_ids=manager.object_ids,
                                                             ignore_label=manager.bg_id)
                avg_iou = manager.frame_level_ious.mean()
                curr_click_count = manager.num_clicks_per_frame.sum()
                metric_compute_end_time = time.perf_counter()
                metric_compute_time.append(metric_compute_end_time - metric_compute_start_time)
                
                dataset_stq[manager.sequence.id].append({
                    "Round": manager.round_num, "#frames": manager.T, "#targets": manager.N, 
                    "#clicks": int(curr_click_count), "IoU": float(avg_iou),
                    "STQ": video_stq, "AQ": video_aq, "SQ": video_sq, 
                })
                logger.info(f"Round {manager.round_num} scores: \n#frames: {manager.T}, \n#targets: {manager.N} \
                    \nTotal #clicks: {curr_click_count} \nIoU: {avg_iou} \
                    \nSTQ: {video_stq} \nAQ: {video_aq} \nSQ: {video_sq}")

                ## WEAKEST PREDICTION ##
                logger.info(f"Looking for an object/frame to refine...")
                # Stopping criterion 1: check whether round budget is over
                if manager.round_num == max_rounds:
                    logger.info(f'Maximum round limit ({max_rounds}) reached!')
                    lowest_frame_index = -1

                # Stopping criterion 2: check whether click budget is over
                if click_budget <= curr_click_count:
                    logger.info(f'Click budget ({max_interactions} per frame) over!')
                    lowest_frame_index = -1

                if lowest_frame_index != -1:                    
                    # select the object with weakest mIoU
                    # also checks stopping criterion 3: check whether all objects meet IoU threshold
                    lowest_frame_index, lowest_obj_id = manager.find_refinement_target(iou_threshold)
                
                if lowest_frame_index != -1:
                    ## CORRECTIVE CLICK ##
                    refined_obj_index = manager.get_corrective_click(frame_idx=lowest_frame_index, obj_id=lowest_obj_id)
                    logger.info(f'Sampled a click on instance {refined_obj_index+1} in frame {lowest_frame_index}')

            logger.info(f"{manager.sequence.id}, time analysis: \
                        \nTotal propagation time: {sum(propagation_time)} \
                        \nAverage propagation time per round: {sum(propagation_time)/len(propagation_time)} \
                        \nTotal Metric computation time: {sum(metric_compute_time)} \
                        \nAverage metric computation time per round: {sum(metric_compute_time)/len(metric_compute_time)}")
            
            del manager
        return dataset_stq


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