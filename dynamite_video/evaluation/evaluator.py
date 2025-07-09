import os
import yaml

from detectron2.utils import comm
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from dynamite_video.data.dataset_builder import build_evaluation_dataset

class Evaluator(DefaultTrainer):
    
    @classmethod
    def interactive_evaluation(cls, cfg, model):
        """
        Perform interactive evaluation on evaluation datasets one-by-one

        Args:
            cfg: experiment configuration
            args: command line arguments
            model: trained model (with weights already loaded)
        """
        
        logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="Evaluator")

        eval_datasets = cfg.ITERATIVE.TEST.DATASETS
        iou_threshold = cfg.ITERATIVE.TEST.IOU_THRESHOLD
        max_interactions = cfg.ITERATIVE.TEST.MAX_INTERACTIONS_PER_TARGET
        max_rounds = cfg.ITERATIVE.TEST.MAX_ROUNDS
        eval_strategy = cfg.ITERATIVE.TEST.EVAL_STRATEGY
        seed_id = cfg.SEED
        save_vis = cfg.ITERATIVE.TEST.SAVE_VISUALIZATIONS

        logger.info("Initiating interactive evaluation with following setup:")
        logger.info(f"Evaluation datasets: {eval_datasets}")
        logger.info(f"IoU threshold: {iou_threshold}")
        logger.info(f"Max #interactions per target: {max_interactions}")
        logger.info(f"Max corrective rounds: {max_rounds}")
        logger.info(f"Evaluation strategy: {eval_strategy}")
        logger.info(f"Random seed: {seed_id}")
        logger.info(f"Output path: {cfg.OUTPUT_DIR}")
        if save_vis:
            logger.info(f"Visualizations saved to: {cfg.OUTPUT_DIR}")

        # Evaluate one dataset at a time
        for dataset_name in eval_datasets:
            
            # load sequence info from disc into `GenericVideoSequence` format
            logger.info(f"Building evaluation dataset from {dataset_name}...")
            dataset, dataset_meta = build_evaluation_dataset(cfg, dataset_name)
            
            from dynamite_video.evaluation.multi_instance_evaluation import evaluate
            
            dataset_result = evaluate(cfg,
                            model,
                            dataset,
                            dataset_meta,
                            iou_threshold=iou_threshold,
                            max_interactions=max_interactions,
                            max_rounds=max_rounds,
                            eval_strategy=eval_strategy,
                            seed_id=seed_id,
                            save_vis=save_vis,
            )
            # save result
            with open(os.path.join(cfg.OUTPUT_DIR, f"metrics_{dataset_name}.yaml"), 'w') as f:
                yaml.dump(dict(dataset_result), f)
        
            for vid, res in dataset_result.items():
                logger.info(f"Video: {vid}")
                logger.info(f"Scores: {res}")