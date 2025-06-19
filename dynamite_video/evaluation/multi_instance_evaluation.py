import numpy as np
import os
import random
import torch
import torch.nn as nn

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from tqdm import tqdm

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.evaluation.manager import SequenceManager
from dynamite_video.evaluation.metrics.metrics import compute_j_and_f, compute_stq


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
    
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="Multi Instance Evaluation")
    logger.info(f"Starting inference on {len(dataset)} sequences...")
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        random.seed(123456+seed_id)

        avg = []

        pbar = tqdm(dataset, leave=False)

        dataset_stq = defaultdict(list)
        
        for i, video in enumerate(pbar):
            pbar.set_description(f"{video.id}")

            # a fresh model for each sequence
            predictor = Predictor(model, len(video))
            
            # sequence manager for current sequence
            manager = SequenceManager(video, dataset_meta, cfg.INPUT)
            
            # click budget for whole sequence
            click_budget = max_interactions * manager.N

            # rounding starts at first frame
            round_num = 1
            lowest_frame_index = 0

            while True: 

                # generate indices of shorter sub-sequences or clips from the whole sequence
                clip_indices = manager.generate_clip_indices(start=lowest_frame_index)

                ## PROPAGATION ##
                for num, indices in enumerate(tqdm(clip_indices, leave=False, desc="Clip")):

                    clip, clip_inputs = manager.extract_non_overlapping_clip(indices)
                    binary_pred_masks = predictor.get_prediction([clip_inputs], indices)    # T,N,H,W
                    panoptic_pred_masks = manager.store_prediction(binary_pred_masks, clip)

                    if save_vis:
                        manager.save_visualization(vis_path, round_num, indices)

                # metrics
                video_stq, video_aq, video_iou = compute_stq(manager.gt_masks, manager.pred_masks, dataset_meta)
                video_j_and_f = compute_j_and_f(manager.gt_masks, manager.pred_masks, manager.N)
                dataset_stq[manager.sequence.id].append({"Round": round_num, "STQ": video_stq, "AQ": video_aq, "IoU": video_iou, "J&F": video_j_and_f.mean()})
                logger.info(f"{manager.sequence.id}, Round {round_num} scores: \nSTQ: {video_stq} \nAQ: {video_aq} \nIoU: {video_iou} \nJ&F: {video_j_and_f.mean()}")

                curr_click_count = ...

                ## WEAKEST PREDICTION ##
                while True:
                    # Stopping criterion 1: check whether round budget is over
                    if round_num == max_rounds:
                        logger.info(f'{manager.sequence.id}, Round {round_num}:: Maximum round limit ({max_rounds}) reached!')
                        lowest_frame_index = -1
                        break

                    # Stopping criterion 2: check whether click budget is over
                    if click_budget == curr_click_count:
                        logger.info(f'{manager.sequence.id}, Round {round_num}:: Click budget ({max_interactions} per frame) over!')
                        lowest_frame_index = -1
                        break

                    # select the object with weakest mIoU and the frame where it has the weakest IoU
                    min_iou, min_iou_index = ...
                    
                    # Stopping criterion 2: check whether all frames meet IoU threshold
                    if min_iou >= iou_threshold:
                        lowest_frame_index = -1
                        logger.info(f'{manager.sequence_id}, Round {round_num}:: All frames meet IoU requirement!')
                        break
                    else:
                        lowest_frame_index = min_iou_index[0]
                        lowest_instance_id = min_iou_index[1]
                
                # hit one of the stopping criteria
                if lowest_frame_index == -1:
                    break

                round_num += 1

                ## CORRECTIVE CLICK ##
                refined_obj_index = manager.get_corrective_click(frame_idx=lowest_frame_index, inst_id=lowest_instance_id)
                logger.info(f'{manager.sequence.id}, Round {round_num}:: Sampled a click on instance {refined_obj_index+1} in frame {lowest_frame_index}')
        
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


class Predictor:
    """
    A wrapper around DynamiteModel interactive evaluation forward pass
    """


    def __init__(self, model, length):
        self.model = model
        self.images = [[] * length]
        self.features = [[] * length]
        self.mask_features = [[] * length]
        self.multi_scale_features = [[] * length]
        self.initialized = False
    
    def get_prediction(self, inputs, indices):
        """
        Args:
            inputs: batched input. Batch size is restricted to 1
        """
        
        if not self.initialized:
            pred_masks, images, features, mask_features, multi_scale_features = self.model(inputs)
            self.initialized = True
        else:
            pred_masks, images, features, mask_features, multi_scale_features = self.model(inputs)

        
        # for i, idx in enumerate(indices):
        #     self.images[idx] = images[i]
        #     self.features[idx] = features[i],
        #     self.mask_features[idx] = mask_features[i]
        #     self.multi_scale_features[idx] = multi_scale_features[i]

        return torch.stack([x.to('cpu',dtype=torch.uint8) for x in pred_masks])