import os
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
            
            dataset_result = evaluate(self.model,
                                    dataset,
                                    dataset_meta,
                                    self.cfg.INPUT,
                                    iou_threshold=self.iou_threshold,
                                    max_interactions=self.max_interactions,
                                    max_rounds=self.max_rounds,
                                    eval_strategy=self.eval_strategy,
                                    min_mask_area=self.min_mask_area,
                                    output_dir=self.output_dir,
                                    seed_id=self.seed_id,
                                    save_vis=self.save_vis
                    )
            # save result
            with open(os.path.join(self.output_dir, f"metrics_{dataset_name}.yaml"), 'w') as f:
                yaml.dump(dict(dataset_result), f)
        
            avg_stq = 0
            avg_aq = 0
            avg_sq = 0
            
            num_vids = 0
            for vid, res in dataset_result.items():
                self.logger.info(f"Video: {vid}")
                self.logger.info(f"Scores: {res}")
                avg_stq += res["STQ"]
                avg_aq += res["AQ"]
                avg_sq += res["SQ"]
                num_vids += 1
            
            self.logger.info(f"All videos: ")
            self.logger.info(f"Average STQ: {avg_stq / num_vids}")
            self.logger.info(f"Average AQ: {avg_aq / num_vids}")
            self.logger.info(f"Average SQ: {avg_sq / num_vids}")