import os
import time
import random
import pickle
import torch
import torch.nn as nn
import numpy as np

from contextlib import ExitStack, contextmanager

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.evaluation.manager import SequenceManager
from dynamite_video.evaluation.metrics import batched_f_measure, batched_jaccard


def evaluate(model, 
             dataset,
             iou_threshold=0.85,
             max_interactions=3,
             max_rounds=3,
             eval_strategy="random",
             seed_id=0,
             output_path=None,
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
    assert output_path is not None, f"output_path not found!"
    vis_path = None
    if save_vis:
        vis_path = os.path.join(output_path, "vis")
        os.makedirs(vis_path, exist_ok=True)
    
    logger = setup_logger(output=output_path, distributed_rank=comm.get_rank(), name="Single Instance Evaluation")
    logger.info(f"Starting inference on {len(dataset)} sequences...")
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        random.seed(123456+seed_id)

        avg = []
        
        for sequence in dataset:

            sequence_instance_ids = sequence["serial_to_orig_ids"].keys()
            sub_sequence_list = []
            manager_list = []

            # a fresh model for each sequence
            predictor = Predictor(model)
            
            for inst_id in sequence_instance_ids:
                
                # instances per frame
                inst_per_frame = []
                for record in sequence["instances_per_frame"]:
                    if inst_id in record:
                        inst_per_frame.append([1])
                    else:
                        inst_per_frame.append([])
                
                sub_sequence = {
                    "id": sequence["id"],
                    "length": sequence["length"],
                    "orig_dims": sequence["orig_dims"],
                    "orig_to_serial_ids": {inst_id: 1},
                    "serial_to_orig_ids": {1: inst_id},
                    "instance_discovery": {1: sequence["instance_discovery"][inst_id]},
                    "images": sequence["images"],
                    "instance_masks": np.expand_dims(sequence["instance_masks"][:, inst_id-1, :, :], 1),
                    "semantic_maps": sequence["instance_masks"][:, inst_id-1, :, :],
                    "instances_per_frame": inst_per_frame,
                    "padding_mask": sequence["padding_mask"],
                    "clip_length": sequence["clip_length"],
                    "num_overlapping_frames": sequence["num_overlapping_frames"],
                }
                sub_sequence_list.append([sub_sequence])
                manager_list.append(SequenceManager(sub_sequence))

            manager = SequenceManager(sequence)
            # ground truth semantic maps [T,H,W] of the sequence frames
            gt_semantic_maps = manager.gt_semantic_maps
            # click budget per frame
            max_iters_for_image = max_interactions * manager.num_instances
            
            round_num = 1
            lowest_frame_index = 0
            while True:
        
                pred_semantic_maps = np.zeros_like(sequence["semantic_maps"])
                
                for inst_id, sub_manager in zip(sequence_instance_ids, manager_list):
                    assert inst_id == list(sub_manager.orig_to_serial_ids.keys())[0]

                    sub_clip_indices = sub_manager.create_clip_indices(start=lowest_frame_index)

                    # forward prediction
                    for sub_indices in sub_clip_indices:
                        # extract a clip with first set of foreground clicks
                        sub_inputs = sub_manager.extract_clip(sub_indices)
                        # obtain predicted instance-wise binary segmentation masks
                        sub_pred_masks = predictor.get_prediction([sub_inputs])
                        sub_manager.store_pred_masks(sub_pred_masks, sub_indices)

                    if save_vis:
                        sub_manager.store_predicted_semantic_maps(sub_pred_masks)
                        sub_manager.save_visualization(vis_path, round_num)
                        
                    pred_semantic_maps[np.where(sub_manager.pred_masks==1)] = inst_id
                
                manager.store_predicted_semantic_maps(pred_semantic_maps)
                
                if save_vis:
                    manager.save_visualization(vis_path, round_num)
                
                # calculate J&F
                jaccard_mean, jaccard_instances = batched_jaccard(gt_semantic_maps, manager.pred_semantic_maps, average_over_objects=True, nb_objects=manager.num_instances)
                contour_mean, _ = batched_f_measure(gt_semantic_maps, manager.pred_semantic_maps, average_over_objects=True, nb_objects=manager.num_instances)
                j_and_f = 0.5*jaccard_mean + 0.5*contour_mean
                logger.info(f'{manager.sequence_id}, Round {round_num}:: Scores: Average IoU: {jaccard_mean.mean()}, Average J&F: {j_and_f.mean()}')

                # find the frame with worst instance-level IoU
                frame_list = np.arange(manager.sequence_length)
                while True:
                    # Stopping criterion 1: check whether round budget is over
                    if round_num == max_rounds:
                        logger.info(f'{manager.sequence_id}, Round {round_num}:: Maximum round limit ({max_rounds}) reached!')
                        lowest_frame_index = -1
                        break

                    # weakest frame and instance
                    min_iou_index = np.unravel_index(np.argmin(jaccard_instances, axis=None), jaccard_instances.shape)
                    min_iou = jaccard_instances[min_iou_index]
                    logger.info(f'{manager.sequence_id}, Round {round_num}:: Next frame to refine: {min_iou_index[0]}, instance: {min_iou_index[1]}, value: {min_iou}')
                    
                    # Stopping criterion 2: check whether all frames meet IoU threshold
                    if min_iou >= iou_threshold:
                        lowest_frame_index = -1
                        logger.info(f'{manager.sequence_id}, Round {round_num}:: All frames meet IoU requirement!')
                        break
                    else:
                        lowest_frame_index = min_iou_index[0]
                        lowest_instance_id = min_iou_index[1]
                        
                        # Check remaining click budget for candidate frame; if not left, find the next weakest frame
                        if manager.num_clicks_per_frame[lowest_frame_index] >= max_iters_for_image:
                            logger.info(f'{manager.sequence_id}, Round {round_num}:: Skipping frame {lowest_frame_index} - click budget over!')
                            np.delete(frame_list, lowest_frame_index)
                            jaccard_instances[min_iou_index[0]] = 99.
                            
                            # Stopping criterion 3: click budget over for all frames
                            if len(frame_list)==0:
                                lowest_frame_index = -1
                                logger.info(f'{manager.sequence_id}, Round {round_num}:: Ran out of click budget for all frames!')
                                break
                        else:
                            break
                
                # hit one of the stopping criteria
                if lowest_frame_index == -1:
                    avg.append([jaccard_mean.mean(), j_and_f.mean()])
                    break

                round_num += 1

                # get corrective clicks
                refined_obj_index = manager.get_corrective_click(frame_idx=lowest_frame_index, inst_id=lowest_instance_id)
                logger.info(f'{manager.sequence_id}, Round {round_num}:: Sampled a click on instance {refined_obj_index+1} in frame {lowest_frame_index}')
        
        return avg


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


    def __init__(self, model):
        self.model = model
        self.images = None
        self.features = None
        self.mask_features = None
        self.multi_scale_features=None
    
    def get_prediction(self, inputs):
        """
        Args:
            inputs: batched input. Batch size is restricted to 1
        """
        
        (pred_masks, _, 
        self.images, _, 
        self.features, 
        self.mask_features,
        self.multi_scale_features, _, _,_) = self.model(inputs)
        # if self.features is None:
        #     # first iteration through the interactive evaluation pipeline 
        #     # generates mask features which is saved to avoid re-computation
        #     (pred_masks, _, 
        #     self.images, _, 
        #     self.features, 
        #     self.mask_features,
        #     self.multi_scale_features, _, _,_) = self.model(inputs)

        # else:
        #     out = self.model()
        #     pred_masks = out[0]

        return [x.to('cpu',dtype=torch.uint8) for x in pred_masks]