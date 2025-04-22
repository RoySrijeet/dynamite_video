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

from dynamite_video.evaluation.clicker import SequenceManager
from dynamite_video.evaluation.metrics import batched_f_measure, batched_jaccard


def interactive_loop():
    ...


def evaluate(model, 
             dataset,
             iou_threshold=0.85,
             max_interactions=3,
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
    
    logger = setup_logger(output=output_path, distributed_rank=comm.get_rank(), name=__name__)
    logger.info(f"Starting inference on {len(dataset)} sequences...")
    
    start_time = time.perf_counter()
    total_data_time = 0 
    total_compute_time = 0

    max_rounds = 2
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        random.seed(123456+seed_id)
        
        for sequence in dataset:
            
            # a fresh model for each sequence
            predictor = Predictor(model)
            manager = SequenceManager(sequence)
            num_instances = len(manager.instances)

            # round 0
            round_num = 0
            for clip_idx, indices in enumerate(sequence["indices"]):
                
                # extract a clip with first set of foreground clicks
                inputs = manager.extract_clip(indices)
                # obtain prediction
                pred_masks = predictor.get_prediction([inputs])

                manager.store_pred_masks(pred_masks, indices)
            
            pred_masks = manager.store_predicted_semantic_maps()
            if save_vis:
                manager.save_visualization(vis_path, round_num)

            max_iters_for_image = max_interactions * num_instances

            jaccard_mean, jaccard_instances = batched_jaccard(manager.gt_semantic_maps, manager.pred_semantic_maps, average_over_objects=True, nb_objects=num_instances)
            contour_mean, contour_instances = batched_f_measure(manager.gt_semantic_maps, manager.pred_semantic_maps, average_over_objects=True, nb_objects=num_instances)

            j_and_f = 0.5*jaccard_mean + 0.5*contour_mean
            j_and_f = j_and_f.tolist()
            seq_avg_jf = sum(j_and_f)/len(j_and_f)

            iou_for_sequence = jaccard_mean.tolist()
            seq_avg_iou = sum(iou_for_sequence)/len(iou_for_sequence)
            print(f'[PROPAGATION INFO][SEQ:{manager.sequence_id}][ROUND:{round_num}] Prediction results: Average IoU: {seq_avg_iou}, Average J&F: {seq_avg_jf}')

            frame_list = [i for i in range(manager.sequence_length)]
            while True:
                min_iou_index = np.unravel_index(np.argmin(jaccard_instances, axis=None), jaccard_instances.shape)
                min_iou = jaccard_instances[min_iou_index]
                print(f'[EVALUATOR INFO][SEQ:{manager.sequence_id}][ROUND:{round_num}] Weakest frame (instance): idx: {min_iou_index}, value: {min_iou}')                        
                if min_iou < iou_threshold:                                                         # 1. whether all frames meet IoU threshold
                    if round_num == max_rounds:                                                     # 2. whether round budget is over
                        print(f'[STOPPING CRITERIA][SEQ:{manager.sequence_id}][ROUND:{round_num}] Maximum round limit ({max_rounds}) reached!')
                        lowest_frame_index = -1
                        break
                    lowest_frame_index = int(min_iou_index[0])
                    print(f'[EVALUATOR INFO][SEQ:{manager.sequence_id}][ROUND:{round_num}] Next index to refine: {lowest_frame_index}, IoU: {min_iou}')
                    break
                    # if num_interactions_for_sequence[lowest_frame_index] >= max_iters_for_image:     # if interaction budget is over for a frame, look for another frame             
                    #     print(f'[STOPPING CRITERIA][SEQ:{manager.sequence_id}][ROUND:{round_num}] Budget over - skipping frame {lowest_frame_index}.')
                    #     frame_list.remove(lowest_frame_index)
                    #     jaccard_instances[min_iou_index[0]] = 99.
                    #     if len(frame_list)==0:                                                         # 3. whether interaction budget is over for all frames
                    #         lowest_frame_index = -1
                    #         print(f'[STOPPING CRITERIA][SEQ:{manager.sequence_id}][ROUND:{round_num}] Ran out of click budget for all frames!')
                    #         break
                    # else:
                    #     break
                else:
                    lowest_frame_index = -1
                    print(f'[STOPPING CRITERIA][SEQ:{manager.sequence_id}][ROUND:{round_num}] All frames meet IoU requirement!')
                    break




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