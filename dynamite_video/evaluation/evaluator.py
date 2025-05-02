import numpy as np

from detectron2.utils import comm
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from dynamite_video.data.dataset_builder import build_evaluation_dataset
from dynamite_video.evaluation.interactive_evaluation import evaluate

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
        eval_strategy = args.eval_strategy
        seed_id = args.seed_id
        save_vis = args.save_vis

        logger.info("Initiating interactive evaluation with following setup:")
        logger.info(f"Evaluation datasets: {eval_datasets}")
        logger.info(f"IoU threshold: {iou_threshold}")
        logger.info(f"Max #interactions: {max_interactions}")
        logger.info(f"Evaluation strategy: {eval_strategy}")
        logger.info(f"Random seed: {seed_id}")
        logger.info(f"Output path: {cfg.OUTPUT_DIR}")
        if save_vis:
            logger.info(f"Visualizations saved to: {cfg.OUTPUT_DIR}")


        # assert iou_threshold >= 0.80

        # Evaluate one dataset at a time
        for dataset_name in eval_datasets:
            
            logger.info(f"Loading dataset: {dataset_name} ...")
            
            # build clips from input dataset
            data = build_evaluation_dataset(cfg, dataset_name, single_instance=cfg.ITERATIVE.TEST.SINGLE_INSTANCE)
            
            result = evaluate(model,
                            data,
                            iou_threshold=iou_threshold,
                            max_interactions=max_interactions,
                            eval_strategy=eval_strategy,
                            seed_id=seed_id,
                            output_path=cfg.OUTPUT_DIR,
                            save_vis=save_vis,
                    )
        
            # save result
            mean_score = np.mean(np.asarray(result), axis=0)
            logger.info(f"Mean scores, {dataset_name}  [IoU, J&F]: {mean_score}")