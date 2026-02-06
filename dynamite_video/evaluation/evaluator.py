import numpy as np
import os
import pandas as pd
import yaml

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.data.dataset_builder import build_evaluation_dataset
from dynamite_video.evaluation.multi_instance_evaluation import evaluate

class Evaluator:

    def __init__(self, cfg):

        self.cfg = cfg
        self.eval_datasets = cfg.ITERATIVE.TEST.DATASETS
        self.iou_threshold = cfg.ITERATIVE.TEST.IOU_THRESHOLD
        self.max_interactions = cfg.ITERATIVE.TEST.MAX_INTERACTIONS_PER_TARGET
        self.max_rounds = cfg.ITERATIVE.TEST.MAX_ROUNDS
        self.eval_strategy = cfg.ITERATIVE.TEST.EVAL_STRATEGY
        self.seed_id = cfg.SEED
        self.save_vis = cfg.ITERATIVE.TEST.SAVE_VISUALIZATIONS
        self.output_dir = cfg.OUTPUT_DIR
        self.min_mask_area = cfg.ITERATIVE.TEST.MIN_MASK_AREA
        self.connected_component_sampling = cfg.ITERATIVE.TEST.CONNECTED_COMPONENT_SAMPLING

        self.logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="Evaluator")
        self.logger.info("Welcome to DynaMITe-Video Evaluation Pipeline!")
        self.logger.info("Initiating interactive evaluation with following setup:")
        self.logger.info(f"Evaluation datasets: {self.eval_datasets}")
        self.logger.info(f"IoU threshold: {self.iou_threshold}")
        self.logger.info(f"Max #interactions per target: {self.max_interactions}")
        self.logger.info(f"Max corrective rounds: {self.max_rounds}")
        self.logger.info(f"Evaluation strategy: {self.eval_strategy}")
        self.logger.info(f"Random seed: {self.seed_id}")
        self.logger.info(f"Output path: {self.output_dir}")
        if self.save_vis:
            self.logger.info(f"Visualizations saved in: {self.output_dir}")

        # build model
        model = build_model(cfg)
        # load model weights
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
        self.model = model

    
    def interactive_evaluation(self):
        """
        Perform interactive evaluation on evaluation datasets one-by-one

        Args:
            cfg: experiment configuration
            args: command line arguments
            model: trained model (with weights already loaded)
        """        

        # Evaluate one dataset at a time
        for dataset_name in self.eval_datasets:
            
            # load sequence info from disc into `GenericVideoSequence` format
            self.logger.info(f"Building evaluation dataset from {dataset_name}...")
            dataset, dataset_meta = build_evaluation_dataset(self.cfg, dataset_name)
            
            dataset_scores, dataset_target_scores, dataset_round_scores = evaluate(self.model,
                                                                                    dataset,
                                                                                    dataset_meta,
                                                                                    self.cfg.INPUT,
                                                                                    iou_threshold=self.iou_threshold,
                                                                                    max_interactions=self.max_interactions,
                                                                                    max_rounds=self.max_rounds,
                                                                                    eval_strategy=self.eval_strategy,
                                                                                    min_mask_area=self.min_mask_area,
                                                                                    connected_component_sampling=self.connected_component_sampling,
                                                                                    output_dir=self.output_dir,
                                                                                    seed_id=self.seed_id,
                                                                                    save_vis=self.save_vis
                                                                            )
            
            #  calculate dataset-level scores
            ds_stq, ds_sq, ds_aq = 0, 0, 0
            ds_sq_per_target = []
            ds_clicks_per_target = []
            ds_rounds, ds_T, ds_N, ds_clicks = 0, 0, 0, 0
            for scores, target_scores in zip(dataset_scores, dataset_target_scores):
                ds_rounds += scores["Round"]
                ds_T += scores["#frames"]
                ds_N += scores["#targets"]
                ds_clicks += scores["#clicks"]
                
                ds_stq += scores["STQ"]
                ds_aq += scores["AQ"]
                ds_sq += scores["SQ"]

                ds_sq_per_target.extend(target_scores["sq_per_target"])
                ds_clicks_per_target.extend(target_scores["num_clicks_per_target"])
            
            ds_sq_per_target = np.asarray(ds_sq_per_target)
            ds_PFO = len(np.where(ds_sq_per_target < self.iou_threshold)[0]) / ds_N
            ds_PMO = len(np.where(ds_sq_per_target == 0)[0]) / ds_N
            ds_NoC = 0
            for obj_id, obj_sq in enumerate(ds_sq_per_target):
                if obj_sq >= self.iou_threshold:
                    ds_NoC += ds_clicks_per_target[obj_id]
                else:
                    ds_NoC += self.max_interactions
            ds_NoC = ds_NoC/ds_N

            entry = {
                "Name": dataset_name,
                "Round": ds_rounds / len(dataset_scores),
                "#frames": ds_T,
                "#targets": ds_N,
                "#clicks": ds_clicks,
                "STQ": ds_stq / ds_N,
                "AQ": ds_aq / ds_N,
                "SQ": ds_sq / ds_N,
                "PFO": ds_PFO,
                "PMO": ds_PMO,
                "NoC": ds_NoC
            }
            dataset_scores.insert(0, entry)
            df = pd.DataFrame(dataset_scores)
            df.to_csv(os.path.join(self.output_dir, f"metrics_{dataset_name}.csv"), index=False)

            # save result
            # with open(os.path.join(self.output_dir, f"metrics_{dataset_name}.yaml"), 'w') as f:
            #     yaml.dump(dict(dataset_scores), f)
        