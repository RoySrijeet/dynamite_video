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
        
        for video in dataset:
            print(f"Processing {video.id}")

            # a fresh model for each sequence
            predictor = Predictor(model, len(video))
            
            manager = SequenceManager(video, dataset_meta, cfg.INPUT)
            
            # round 1
            round_num = 1
            lowest_frame_index = 0

            # generate indices of shorter sub-sequences or clips from the whole sequence
            clip_indices = manager.generate_clip_indices(start=lowest_frame_index)

            # make predictions for one clip at a time
            for num, indices in enumerate(clip_indices):

                clip, clip_inputs = manager.extract_clip(indices)
                clip_preds = predictor.get_prediction([clip_inputs], indices)    # T,N,H,W
                manager.store_prediction(clip_preds, clip, indices)

                manager.save_visualization(vis_path=vis_path, round_num=1, indices=indices)

            del manager

            continue
            
            

            # ground truth semantic maps [T,H,W] of the sequence frames
            gt_semantic_maps = manager.gt_semantic_maps
            # click budget per frame
            max_iters_for_image = max_interactions * manager.num_instances
            
            ####### Rounds #######
            # 1. Obtain predicted masks across the whole sequence
            # 2. Find the frame with the worst instance segmentation map
            # 3. Get corrective clicks on that frame/instance
            # Repeat
            ######################

            # round 1 starts from the first frame
            round_num = 1
            lowest_frame_index = 0
            while True:
                
                # generate indices of shorter clips from whole sequence
                clip_indices = manager.create_clip_indices(start=lowest_frame_index)
                
                # TODO: propagation cut-off
                
                # forward prediction
                for indices in clip_indices:
                    # extract a clip with first set of foreground clicks
                    inputs = manager.extract_clip(indices)
                    # obtain predicted instance-wise binary segmentation masks
                    pred_masks = predictor.get_prediction([inputs])
                    manager.store_pred_masks(pred_masks, indices)
                
                # convert predicted binary masks to semantic maps
                manager.store_predicted_semantic_maps()
                
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

        return [x.to('cpu',dtype=torch.uint8) for x in pred_masks]