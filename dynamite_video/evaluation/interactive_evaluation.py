import time
import random
import torch
import torch.nn as nn

from contextlib import ExitStack, contextmanager

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.evaluation.clicker import SequenceManager

def evaluate(model, 
             dataset,
             iou_threshold=0.85,
             max_interactions=3,
             eval_strategy="random",
             seed_id=0,
             output_path=None,
             vis_path=None
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
        vis_path: path to save visualization of masks with corrective clicks (str, default None)
    """
    assert output_path is not None, f"output_path not found!"
    
    logger = setup_logger(output=output_path, distributed_rank=comm.get_rank(), name=__name__)
    logger.info(f"Starting inference on {len(dataset)} sequences...")
    
    start_time = time.perf_counter()
    total_data_time = 0 
    total_compute_time = 0
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        random.seed(123456+seed_id)
        
        for sequence in dataset:
            
            # a fresh model for each sequence
            predictor = Predictor(model)
            manager = SequenceManager(sequence)

            for indices in sequence["indices"]:
                
                # extract a clip
                inputs = manager.extract_clip(indices)
                # obtain prediction
                pred_masks = predictor.get_prediction([inputs])

                manager.save_pred_masks(pred_masks, indices)

            
            del predictor, manager






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
        
        if self.features is None:
            # first iteration through the interactive evaluation pipeline 
            # generates mask features which is saved to avoid re-computation
            (pred_masks, _, 
            self.images, _, 
            self.features, 
            self.mask_features,
            self.multi_scale_features, _, _,_) = self.model(inputs)

        else:
            out = self.model()
            pred_masks = out[0]

        return pred_masks.to('cpu',dtype=torch.uint8)