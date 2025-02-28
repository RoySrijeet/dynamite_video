from detectron2.engine import DefaultTrainer

from dynamite_video.evaluation.interactive_evaluation import evaluate

class Evaluator(DefaultTrainer):

    # def __init__(self, cfg):
    #     self.cfg = cfg

    @classmethod
    def build_test_loader(cls, dataset_name):
        ...
    
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

            dataloader = cls.build_test_loader(cfg, dataset_name)
            result = evaluate()
        
        # save result - TODO
