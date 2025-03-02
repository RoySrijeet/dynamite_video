import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from dynamite_video.data.datasets import *

from dynamite_video.utils.paths import Paths

DATASET_BUILDERS = {
    "BURST": BURSTTrainingDataset,
    "CITYSCAPES_VPS": CITYSCAPESVPSTrainingDataset,
    "DAVIS": DAVISTrainingDataset,
    "KITTI_STEP": KITTISTEPTrainingDataset,
    "MOSE": MOSETrainingDataset,
    "PUMAVOS": PUMAVOSTrainingDataset,
    "VIPSEG": VIPSEGTrainingDataset,
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

    if len(dataset_list) > 1:
        dataset_num_samples = np.round(np.array(dataset_weights, np.float32) * num_samples).astype(int)
        dataset_num_samples[-1] = num_samples - dataset_num_samples[:-1].sum()
        dataset_num_samples = dataset_num_samples.tolist()
    else:
        dataset_num_samples = [num_samples]
    
    datasets = []
    for ds_name, ds_num_samples in zip(dataset_list, dataset_num_samples):
        datasets.append(DATASET_BUILDERS[ds_name](cfg, ds_num_samples))

    if True:
        return listify(datasets)

    if len(datasets) > 1:
        return ConcatDataset(datasets)
    else:
        return datasets[0]
    

def listify(datasets):
    dataset_list = []
    for ds in datasets:
        dataset_list.extend(ds.samples)
    return dataset_list