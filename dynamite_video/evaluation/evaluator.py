from detectron2.engine import DefaultTrainer

from dynamite_video.data.dataset_builder import build_evaluation_dataset
from dynamite_video.evaluation.interactive_evaluation import evaluate

class Evaluator(DefaultTrainer):

    # def __init__(self, cfg):
    #     self.cfg = cfg
    
    @classmethod
    def interactive_evaluation(cls, cfg, args):
        
        eval_datasets = args.eval_datasets
        vis_path = args.vis_path
        seed_id = args.seed_id
        iou_threshold = args.iou_threshold
        max_interactions = args.max_interactions

        # assert iou_threshold >= 0.80

        # NOTE - evaluation only supported for a handful of datasets
        for dataset_name in eval_datasets:

            eval_dataset = build_evaluation_dataset(cfg, dataset_name)
            result = evaluate(eval_dataset)
        
        # save result - TODO