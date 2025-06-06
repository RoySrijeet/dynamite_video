import numpy as np

from detectron2.utils import comm
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from dynamite_video.data.dataset_builder import build_evaluation_dataset

class Evaluator(DefaultTrainer):
    
    @classmethod
    def interactive_evaluation(cls, cfg, args, model):
        """
        Perform interactive evaluation on evaluation datasets one-by-one

        Args:
            cfg: experiment configuration
            args: command line arguments
            model: trained model (with weights already loaded)
        """
        
        logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="Evaluator")

        eval_datasets = args.eval_datasets
        iou_threshold = args.iou_threshold
        max_interactions = args.max_interactions
        max_rounds = args.max_rounds
        eval_strategy = args.eval_strategy
        seed_id = args.seed_id
        save_vis = args.save_vis

        logger.info("Initiating interactive evaluation with following setup:")
        logger.info(f"Evaluation datasets: {eval_datasets}")
        logger.info(f"IoU threshold: {iou_threshold}")
        logger.info(f"Max #interactions: {max_interactions}")
        logger.info(f"Max corrective rounds: {max_rounds}")
        logger.info(f"Evaluation strategy: {eval_strategy}")
        logger.info(f"Random seed: {seed_id}")
        logger.info(f"Output path: {cfg.OUTPUT_DIR}")
        if save_vis:
            logger.info(f"Visualizations saved to: {cfg.OUTPUT_DIR}")

        # Evaluate one dataset at a time
        for dataset_name in eval_datasets:
            
            # load sequence info from disc into `GenericVideoSequence` format
            dataset, dataset_meta = build_evaluation_dataset(cfg, dataset_name)
            
            from dynamite_video.evaluation.multi_instance_evaluation import evaluate
            
            result = evaluate(cfg,
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
            mean_score = np.mean(np.asarray(result), axis=0)
            logger.info(f"Mean scores, {dataset_name}  [IoU, J&F]: {mean_score}")