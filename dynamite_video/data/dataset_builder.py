import numpy as np

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.data.datasets import *

TRAINING_DATASET_BUILDERS = {
    "ADE20K": ADE20KPanopticDataset,
    "COCO": COCOPanopticDataset,
    "BURST": BURSTTrainingDataset,
    "DAVIS": DAVISTrainingDataset,
    "KITTI_STEP": KITTISTEPTrainingDataset,
    "VIPSEG": VIPSEGTrainingDataset,
}

EVALUATION_DATASET_BUILDERS = {
    "KITTI_STEP": KITTISTEPEvaluationDataset,
    "DAVIS": DAVISEvaluationDataset,
    "VIPSEG": VIPSEGEvaluationDataset,
}

def build_training_dataset(cfg):
    """
    Load training samples from one or more datasets and return as a single training dataset
    """
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)
    if cfg.TRAINING.PRETRAIN:
        dataset_list = cfg.TRAINING.PRETRAIN_DATASET_LIST
        dataset_weights = cfg.TRAINING.PRETRAIN_DATASET_WEIGHTS
    else:
        dataset_list = cfg.TRAINING.DATASET_LIST
        dataset_weights = cfg.TRAINING.DATASET_WEIGHTS
    total_iterations = cfg.SOLVER.MAX_ITER
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    num_samples = total_iterations * batch_size
    
    logger.info(f"Building training dataset from following datasets: {dataset_list}")
    logger.info(f"Building training dataset with following dataset weights: {dataset_weights}")
    logger.info(f"Number of training samples: {num_samples} (MAX_ITER: {total_iterations}, BATCH_SIZE: {batch_size})")
    
    if len(dataset_list) > 1:
        dataset_num_samples = np.round(np.array(dataset_weights, np.float32) * num_samples).astype(int)
        dataset_num_samples[-1] = num_samples - dataset_num_samples[:-1].sum()
        dataset_num_samples = dataset_num_samples.tolist()
    else:
        dataset_num_samples = [num_samples]
    
    datasets = []
    for ds_name, ds_num_samples in zip(dataset_list, dataset_num_samples):
        logger.info(f"Loading samples from {ds_name}...")
        datasets.append(TRAINING_DATASET_BUILDERS[ds_name](cfg, ds_num_samples))    
    logger.info(f"Done!")

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        return datasets[0]


def build_evaluation_dataset(cfg, dataset_name):
    """
    Load evaluation dataset in clips
    """
    eval_ds = EVALUATION_DATASET_BUILDERS[dataset_name](cfg)
    
    return eval_ds.videos, eval_ds.meta