import os
import copy
import time
import torch
import logging
import itertools
import torch.nn as nn
from datetime import datetime, timedelta

from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.utils.data import DataLoader

from data.utils.collate import Collator
from data.utils.sampler import DistributedSampler
from data.dataset_builder import build_training_dataset

from dynamite_video.training.trainer import TrainerBase
import dynamite_video.training.utils.logger as LogManager
from dynamite_video.training.utils.timer import ETAEstimator
from dynamite_video.training.utils.checkpointer import CheckpointManager
from dynamite_video.training.utils.wandb_logger import WeightsAndBiasesLogger
from dynamite_video.training.utils.tensorboard_logger import TensorboardLogger
from dynamite_video.training.utils.interrupt_detector import InterruptDetector

import dynamite_video.utils.distributed as dist_utils

class Trainer(TrainerBase):
    """
    Extension of the Trainer class adapted to Mask2Former
    """
    
    def __init__(
        self,
        cfg,
        model: nn.Module,
        output_dir: str,
        log_dir: str,
        console_logger: logging.Logger,
        metrics_logger: Union[WeightsAndBiasesLogger, TensorboardLogger],
        eta_estimator: ETAEstimator,
        checkpoint_manager: CheckpointManager,
        grad_scaler: Union[torch.amp.GradScaler, None],
        state_params: Dict[str, Any]
    ):
        super().__init__(cfg)

        self.model = model
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.console_logger = console_logger

        self.optimizers_and_lr_schedulers = []
        self.optimizer = self.build_optimizer(self.cfg, self.model)
        # self.create_optimizers(self._model) #TODO - enable after model
        # TarViS's create_optimizers adds the optimizer as optimizers and lr scheduler property
        # DynaMITe returns the optimizer

        self.metrics_logger = metrics_logger
        self.eta_estimator = eta_estimator
        self.checkpoint_manager = checkpoint_manager
        self.interrupt_detector = InterruptDetector()
        self.grad_scaler = grad_scaler
        self._state_params = state_params

        self.logging_buffer = dict()
        self.current_session_start_time = time.time()
        self.detect_anomaly = False
        self.ignore_oom_errors = False
        self.log_time_durations = False

        self.display_interval = 1
        self.summary_interval = -1
        self.image_summary_interval = -1

        #self.setup()

    @property
    def _model(self):
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
    
    @property
    def total_iterations(self):
        return self._state_params["total_iterations"]

    @property
    def elapsed_iterations(self):
        return self._state_params["elapsed_iterations"]
    
    @property
    def num_gpus(self):
        return dist_utils.get_world_size()
    
    @property
    def local_rank(self):
        return dist_utils.get_local_rank()

    @property
    def global_rank(self):
        return dist_utils.get_rank()

    @property
    def local_device(self):
        return dist_utils.get_device()
    
    @staticmethod
    def get_num_available_cpu_cores() -> int:
        # When running under SLURM, we need to check the `SLURM_CPUS_PER_TASK` environment variable to get the
        # correct number of available CPU cores because multiprocessing.cpu_count() just returns the total number of
        # CPU cores on the machine regardless of how many SLURM has allocated for the given job.
        total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        return min(8, max(1, total_cores // local_world_size))
    
    
    def load_model_weights(self, state_dict: Dict[str, torch.Tensor]):
        self._model.load_state_dict(state_dict, strict=True)

    
    def print(self, msg, *args, **kwargs):
        level = logging.getLevelName(kwargs.pop("level", "INFO"))
        assert isinstance(level, int), f"Given log level '{level}' is invalid"
        self.console_logger.log(level, msg, *args, **kwargs)


    @classmethod
    def new(
        cls,
        cfg, 
        model: nn.Module,
        output_dir: str,
        total_iterations: int,
        save_interval: int,
        restore_model_weights: Optional[str] = None,
        use_mixed_precision: Optional[bool] = False,
        find_model_unused_parameters: Optional[bool] = True,
        convert_sync_batchnorm: Optional[bool] = False,
        start_saving_checkpoints_after: Optional[int] = 0,
        max_checkpoints_to_keep: Optional[int] = -1,
        wandb_logging: Optional[bool] = False,
        wandb_project: Optional[str] = None,
        wandb_run: Optional[str] = None,
        wandb_config_dict: Optional[Dict[str, Any]] = None,
        console_logger: Optional[logging.Logger] = None,
        max_runtime: Optional[timedelta] = timedelta(days=1000)
    ):
        # create log directory
        log_dir = os.path.join(output_dir, "logs")
        if dist_utils.is_main_process():
            os.makedirs(log_dir, exist_ok=True)
        # wait until main process has created the log directory
        dist_utils.synchronize() 
        
        # initialize console logger
        if console_logger is None:
            if dist_utils.is_main_process():
                log_txt_file = os.path.join(log_dir, "out.log")
            else:
                log_txt_file = os.path.join(log_dir, f"out_rank{dist_utils.get_rank()}.log")
            console_logger = LogManager.create_console_logger(logging.INFO, logging.WARN, file_output_path=log_txt_file)
        else:
            console_logger = console_logger

        assert start_saving_checkpoints_after < total_iterations
        assert save_interval < total_iterations

        # ETA estimator
        eta_estimator = ETAEstimator.create(
            total_iterations=total_iterations,
            num_iterations_to_discard=50
        )

        # Checkpoint manager
        checkpoint_manager = CheckpointManager.create(
            logger=console_logger,
            checkpoint_dir=output_dir,
            save_interval=save_interval,
            start_saving_after=start_saving_checkpoints_after,
            max_num_to_keep=max_checkpoints_to_keep
        )

        # WeightsAndBiases/Tensorboard logger
        if wandb_logging:
            assert wandb_project is not None
            metrics_logger = WeightsAndBiasesLogger.create(
                project=wandb_project,
                run_name=wandb_run,
                config=wandb_config_dict,
                suppress_console_output=True,
                suppress_failure=False
            )
        else:
            metrics_logger = TensorboardLogger(
                output_dir=os.path.join(log_dir, "tensorboard")
            )

        # Gradient scaler for mixed precision
        gradient_scaler = None
        if use_mixed_precision:
            gradient_scaler = torch.amp.GradScaler("cuda")

        # Wrap model with DDP
        if dist_utils.is_distributed():
            if convert_sync_batchnorm:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # For multi-node training, it is important to use the local rank here
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[dist_utils.get_local_rank()], output_device=dist_utils.get_local_rank(),
                find_unused_parameters=find_model_unused_parameters
            )

        # max run-time
        max_runtime = (max_runtime.days * 3600 * 24) + max_runtime.seconds

        trainer_state_params = {
            "find_model_unused_parameters": find_model_unused_parameters,
            "convert_sync_batchnorm": convert_sync_batchnorm,
            "total_iterations": total_iterations,
            "elapsed_iterations": 0,
            "max_runtime": int(max_runtime)
        }

        trainer = cls(
            cfg,
            model=model,
            output_dir=output_dir,
            log_dir=log_dir,
            console_logger=console_logger,
            metrics_logger=metrics_logger,
            eta_estimator=eta_estimator,
            checkpoint_manager=checkpoint_manager,
            grad_scaler=gradient_scaler,
            state_params=trainer_state_params,
        )

        if restore_model_weights:
            checkpoint_data = torch.load(restore_model_weights, map_location=dist_utils.get_device())
            trainer.load_model_weights(checkpoint_data["model"])
            console_logger.info(f"Restored model weights from {restore_model_weights}")
        
        return trainer

    def calculate_optimizer_step_parameters(self, accumulate_gradients: bool, batch_size: int,
                                            max_samples_per_gpu: int) -> Tuple[int, int]:
        if accumulate_gradients:
            # ensure that batch size is larger than the number of available GPUs
            assert batch_size >= self.num_gpus, f"Batch size ({batch_size}) must be >= number of GPUs ({self.num_gpus})"

            if batch_size < (max_samples_per_gpu * self.num_gpus):
                # we have more GPUs than needed
                assert batch_size % self.num_gpus == 0, \
                    f"Batch size ({batch_size}) must be exactly divisible by number of GPUs ({self.num_gpus})"
                optimizer_step_interval = 1
            else:
                assert batch_size % min(batch_size, max_samples_per_gpu) == 0
                optimizer_step_interval = int(batch_size / (min(batch_size, max_samples_per_gpu) * self.num_gpus))

            assert optimizer_step_interval > 0, \
                f"Oops! Something went wrong. Given params: batch_size={batch_size}, " \
                    f"max_samples_per_gpu={max_samples_per_gpu}, num_gpus={self.num_gpus}"

            self.print(f"Optimizer will be run every {optimizer_step_interval} iterations")

        else:
            if batch_size > (self.num_gpus * max_samples_per_gpu):
                raise ValueError(
                    f"A batch size of {batch_size} cannot be achieved because max "
                    f"samples per GPU = {max_samples_per_gpu} and num GPUs = {self.num_gpus} (product of the two is "
                    f"less than batch size)"
                )
            optimizer_step_interval = 1

        sub_iter_batch_size_per_gpu = batch_size // (optimizer_step_interval * self.num_gpus)

        assert 0 < sub_iter_batch_size_per_gpu <= batch_size, \
            f"Oops! Something went wrong. Given params: batch_size={batch_size}, " \
            f"max_samples_per_gpu={max_samples_per_gpu}, num_gpus={self.num_gpus}," \
            f"optimizer_step_interval={optimizer_step_interval}"

        return sub_iter_batch_size_per_gpu, optimizer_step_interval
    

    @classmethod
    def resume(
        cls,
        cfg,
        model: nn.Module,
        checkpoint_path: str,
    ):
        """
        Resume training the model from provided checkpoint

        Args:
            cfg
            model: nn.Module model
            checkpoint_path: path to the checkpoint (.pth) file
        """
        print(f"Resume model training from {checkpoint_path}")
        raise NotImplementedError


    
    @classmethod
    def tune(cls):
        ...

    
    def start(
        self,
        batch_size: int,
        accumulate_gradients: bool,
        clip_gradients: bool,
        max_samples_per_gpu: int,
        display_interval: int,
        summary_interval: int,
        image_summary_interval: int = -1,
        force_end_after_total_iterations_elapsed: bool = True,
        dataloader_cpu_workers: int = -1,
        dataloader_collate_fn: Optional[Callable] = None
    ):

        sub_iter_batch_size_per_gpu, optimizer_step_interval = self.calculate_optimizer_step_parameters(
            accumulate_gradients, batch_size, max_samples_per_gpu
        )

        self.display_interval = display_interval
        self.summary_interval = summary_interval
        self.image_summary_interval = image_summary_interval

        # load dataset
        dataset = build_training_dataset(self.cfg)

        # # distributed sampler
        # sampler = DistributedSampler(
        #     dataset, 
        #     self.num_gpus, 
        #     self.global_rank, 
        #     shuffle=True
        # )
        
        collate_fn_train = Collator(self.cfg, is_train=True)
        
        # create training dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4, #self.batch_size
            num_workers=0, #self.get_num_available_cpu_cores()
            collate_fn=collate_fn_train,
        )
        print(len(dataloader))
        _iter = 0
        for batch in dataloader:
            print(_iter)
            _iter += 1
            import pickle
            with open(f"/home/roy/REPOS/dynamite_video/debug/storage/training_batch/batch_{_iter}.pkl", "wb") as f:
                pickle.dump(batch, f)