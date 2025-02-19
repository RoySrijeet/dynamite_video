from typing import Any, Dict, List
from dynamite_video.data.utils.clicker import get_clicks_coords


class Collator:
    def __init__(self, cfg, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
    
    def __call__(self, samples: List[Dict[str, Any]]):
        if self.is_train:
            return collate_fn_train(samples, self.cfg)
        else:
            return collate_fn_inference(samples)
        

def collate_fn_train(batch: List[Dict[str, Any]], cfg):
    """
    Collate function for training data loader. Main purpose is to add clicks.
    
    Args:
        samples: list of samples
        cfg: experiment config
    """
    new_batch = []
    for sample in batch:
        
        # add clicks
        num_clicks_per_object, fg_coords_list, bg_coords_list, max_timestamp_list = get_clicks_coords(
                                                                                        instance_ids=sample["instance_ids"],
                                                                                        instance_masks=sample["instance_masks"], 
                                                                                        bg_masks=sample["bg_masks"],
                                                                                        frame_instance_occupancy=sample["frame_instance_occupancy"],
                                                                                        max_num_points=cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                                                        first_click_center=cfg.CLICKER.TRAINING.FIRST_CLICK_CENTER,
                                                                                        optional_frames_fg_prob=cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                                                        bg_prob=cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                                                        gamma=cfg.CLICKER.TRAINING.GAMMA,
                                                                                        start_t=1,
                                                                                    )


        sample["num_clicks_per_object"] = num_clicks_per_object
        sample["fg_coords_list"] = fg_coords_list
        sample["bg_coords_list"] = bg_coords_list
        sample["max_timestamp_list"] = max_timestamp_list
        new_batch.append(sample)
    
    return new_batch


def collate_fn_inference(samples):
    raise NotImplementedError