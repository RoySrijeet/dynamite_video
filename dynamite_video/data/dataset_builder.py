import numpy as np

from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from dynamite_video.data.datasets import *

TRAINING_DATASET_BUILDERS = {
    "BURST": BURSTTrainingDataset,
    "CITYSCAPES_VPS": CITYSCAPESVPSTrainingDataset,
    "DAVIS": DAVISTrainingDataset,
    "KITTI_STEP": KITTISTEPTrainingDataset,
    "MOSE": MOSETrainingDataset,
    "PUMAVOS": PUMAVOSTrainingDataset,
    "VIPSEG": VIPSEGTrainingDataset,
}

EVALUATION_DATASET_BUILDERS = {
    "DAVIS": DAVISInferenceDataset,
    "BURST": BURSTInferenceDataset,
    "KITTI_STEP": KITTISTEPInferenceDataset,
}

def build_training_dataset(cfg):
    """
    Load training samples from one or more datasets and return as a single training dataset
    """

    dataset_list = cfg.TRAINING.DATASET_LIST
    dataset_weights = cfg.TRAINING.DATASET_WEIGHTS
    total_iterations = cfg.SOLVER.MAX_ITER
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    num_samples = total_iterations * batch_size

    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)
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

    return listify(datasets)
    

def listify(datasets):
    dataset_list = []
    for ds in datasets:
        dataset_list.extend(ds.samples)
    return dataset_list


def build_evaluation_dataset(cfg, dataset_name, single_instance=False):
    """
    Load evaluation dataset in clips
    """
    eval_ds = EVALUATION_DATASET_BUILDERS[dataset_name](cfg)
    return eval_ds.create_inference_dataset(single_instance)