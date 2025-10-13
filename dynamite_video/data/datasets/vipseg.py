import os
import json
import numpy as np
import pycocotools.mask as mt

from PIL import Image
from typing import Dict

from dynamite_video.data.datasets.base import TrainingDataset, EvaluationDataset
from dynamite_video.data.generic_video_parser import GenericVideoSequence, parse_generic_video_dataset
from dynamite_video.data.utils.data_utils import compute_resized_dims, resize_images, resize_masks
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths

class VIPSEGTrainingDataset(TrainingDataset):
    """
    VIPSEG Training Dataset Class

    Creates a `torch.utils.data.Dataset` class to load VIPSEG dataset
    """
    
    def __init__(self, cfg, num_samples: int):
        """
        Read the video frames and their annotations from disc

        Args:
            cfg: configuration
            num_samples: number of samples to be used from this dataset
        """
        
        # number of frames in each training sample
        clip_length = cfg.TRAINING.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.VIPSEG.TRAINING.FPS
        # given a starting frame of a clip (a training sample), this value defines the span of the window 
        # in front of the starting frame, from where the rest of the frames of the clip are to be sampled
        frame_sampling_multiplicative_factor = cfg.DATASETS.VIPSEG.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR

        super().__init__(cfg, "VIPSEG", clip_length, num_samples, fps, frame_sampling_multiplicative_factor)

        # path to VIPSEG images
        self.path_to_images = Paths.to_vipseg_images()
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/vipseg.fpack"
            assert os.path.exists(path_to_images), f"VIPSEG images not found at: {self.path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)

        # training video info
        annotations_content = self.map_annotations(Paths.to_vipseg_annotations(), 
                                                   Paths.to_vipseg_train_video_info())
        
        # cast each video sequence in the dataset to a generic `GenericVideoSequence` template
        videos, meta_info = parse_generic_video_dataset(self.path_to_images, annotations_content)
        
        self.meta = meta_info
        self.videos: Dict[str, GenericVideoSequence] = {vid.id: vid for vid in videos}

        # create samples
        self.samples = self.create_training_samples(self.videos, num_samples)

        self.sample_image_dims = [[1, 1] for _ in range(num_samples)]

        self.annotations_content = annotations_content


    def map_annotations(
            self,
            path_to_annotations: str, 
            video_info_path: str
    ):
        """
        Read VIPSEG annotations
        """

        # only allow instances with mask area above a certain threshold
        MIN_MASK_AREA = self.cfg.TRAINING.MIN_MASK_AREA
        IGNORE_CLASSES = self.cfg.DATASETS.VIPSEG.IGNORE_CLASSES  # [0]
        MAX_INSTANCES_PER_CATEGORY = self.cfg.DATASETS.VIPSEG.MAX_INSTANCES_PER_CATEGORY    # 100
        
        with open(video_info_path, "r") as f:
            video_info = json.load(f)

        # Convention: For panoptic masks, category IDs range from 1 to 124. "0" denotes VOID class. 
        # For "stuff" classes, the value of masks is the same as the category ID. 
        # For "thing" classes, the value of masks is "category_id x 100 + instance_id". 
        # Example - values of masks of "person" (category ID = 61) instances are "6100", "6101", ...

        # NOTE: In JSON file, category IDs range from 0-123 and excludes VOID class. However, the values 
        # in the masks still correspond to original convention. So, the category IDs are shifted (+1) to 
        # match the original convention.
        
        # store category labels in meta info
        categories = {}
        # keep a separate record of IDs of "thing" classes
        things_list = []
        for entry in video_info["categories"]:
            categories[int(entry['id'])+1] = entry['name']
            if entry["isthing"]:
                things_list.append(int(entry['id'])+1)
        meta_info = {
            "category_labels": categories,
            "num_classes": len(categories),
            "things_list": things_list,
            "ignore_class": IGNORE_CLASSES,
            "max_instances_per_category": MAX_INSTANCES_PER_CATEGORY,
            "min_mask_area": MIN_MASK_AREA,
        }
        
        # read annotations
        sequence_annotations = []
        for seq in video_info["sequences"]:
            # each sequence dict has the following keys: 'name', 'filenames', 'stuff_classes', 'thing_classes', 'instance_ids', 'frame_instance_occupancy', 'height', 'width'
            
            entry = {}
            entry["id"] = f"{self.name}_{seq['name']}"
            entry["dataset"]  = self.name
            entry["height"] = seq["height"]
            entry["width"] = seq["width"]
            
            # path to image files
            entry["image_paths"] = sorted([os.path.join(self.path_to_images, seq["name"], file + '.jpg') for file in seq["filenames"]])
            # mask files
            maskfiles = sorted([os.path.join(path_to_annotations, seq["name"], file + '.png') for file in seq["filenames"]])
            
            segmentations = []
            accepted_stuff_classes = set()
            accepted_thing_ids = set()
            for idx, file in enumerate(maskfiles):
                # read panoptic mask
                mask = np.asarray(Image.open(file))
                
                # instances of both "thing" and "stuff" classes
                instances = list(np.unique(mask))
                # remove ignore classes
                instances = [cls_id for cls_id in instances if cls_id not in IGNORE_CLASSES]

                binary_masks = {}
                for inst_id in instances:
                    # extract binary mask
                    _m = (mask==inst_id).astype(dtype='uint8')
                    # check mask area
                    if _m.sum() >= MIN_MASK_AREA:
                        # encode mask array as RLE
                        binary_masks[int(inst_id)] = mt.encode(np.asfortranarray(_m))
                        if (inst_id-1) in seq["stuff_classes"]:
                            accepted_stuff_classes.add(int(inst_id))
                        else:
                            # instance belongs to a "thing" class
                            accepted_thing_ids.add(int(inst_id))
                
                segmentations.append(binary_masks)
            
            entry["segmentations"] = segmentations
            entry["stuff_classes"] = list(accepted_stuff_classes)
            # assigned_id = category_id x MAX_INSTANCES_PER_CATEGORY (100) + instance_id
            entry["thing_classes"] = list(set([k//MAX_INSTANCES_PER_CATEGORY for k in accepted_thing_ids]))
            entry["categories"] = {k:k//MAX_INSTANCES_PER_CATEGORY for k in accepted_thing_ids} # of "thing" classes
            entry["categories"].update({k:k for k in accepted_stuff_classes})

            sequence_annotations.append(entry)

        return {
            "sequences": sequence_annotations,
            "meta": meta_info
        }


########################### EVALUATION DATASET ###########################


class VIPSEGEvaluationDataset(EvaluationDataset):
    """
    Evaluation dataset for VIPSEG ("val" or "test" split)
    """

    def __init__(self, cfg):
        # number of frames in each training sample
        clip_length = cfg.DATASETS.VIPSEG.EVALUATION.CLIP_LENGTH
        # video fps
        fps = cfg.DATASETS.VIPSEG.EVALUATION.FPS
        # number of overlapping frames between clips
        num_overlapping_frames = cfg.DATASETS.VIPSEG.EVALUATION.FRAME_OVERLAP

        assert num_overlapping_frames <= clip_length, f"No. of overlapping frames cannot be more than the length of a clip"
        
        split = cfg.DATASETS.VIPSEG.EVALUATION.SPLIT
        assert split in ["val", "test"]
        
        super().__init__(cfg, "VIPSEG", clip_length, fps, num_overlapping_frames, split)

        # get paths
        self.path_to_images = Paths.to_vipseg_images()
        
        self.path_to_annotations = Paths.to_vipseg_annotations()
        if self.split == "val":
            self.path_to_val_imset = Paths.to_vipseg_val_imset()
        else:
            self.path_to_val_imset = Paths.to_vipseg_test_imset()
        
        
    def create_evaluation_dataset(self):
        """
        Prepare dataset for evaluation.
        """
        
        # evaluation sequences
        with open(self.path_to_val_imset, "r") as f:
            sequences = [seq.rstrip() for seq in f.readlines()]

        sequence_annotations = []
        for seq in sequences:

            metadata = {"id": seq}

            # load images
            image_filepaths = sorted([os.path.join(self.path_to_images, seq, file) for file in os.listdir(os.path.join(self.path_to_images, seq)) if file.endswith('jpg')])
            images = self.load_images(image_filepaths)      # [T, H, W, 3]

            metadata["length"] = len(image_filepaths)
            metadata["orig_dims"] = (images.shape[1], images.shape[2])

            # panoptic masks
            mask_filepaths = sorted([os.path.join(self.path_to_annotations, seq, file) for file in os.listdir(os.path.join(self.path_to_annotations, seq)) if file.endswith('png')])
            semantic_maps, bg_masks, instances_per_frame, instance_discovery = self.load_png_masks(mask_filepaths)

            # resize
            if self.cfg.INPUT.AUGMENTATION.RESIZE_TEST:
                # compute target resolution
                new_height, new_width = compute_resized_dims(
                    *images.shape[1:3], 
                    min_dim=self.cfg.INPUT.AUGMENTATION.MIN_DIM_TEST,
                    max_dim=self.cfg.INPUT.AUGMENTATION.MAX_DIM_TEST,
                )
                if (new_height, new_width) != metadata["orig_dims"]:
                    images = resize_images(images, new_height, new_width)
                    semantic_maps = resize_masks(semantic_maps, new_height, new_width, binary=False)
                    bg_masks = resize_masks(bg_masks, new_height, new_width, binary=False)
            
            # arrange dimensions
            images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]
            if self.cfg.INPUT.RGB:
                # BGR -> RGB (load_images uses cv2.imread which reads images in BGR mode by default)
                images = np.flip(images, 1).copy()
            
            metadata["images"] = images
            metadata["bg_masks"] = bg_masks
            # TODO - padding - not applied
            metadata["padding_mask"] = np.zeros((images.shape[2], images.shape[3])).astype('uint8')

            # serialize instance IDs
            # IDs of all instances present in the sequence
            orig_instance_ids = sorted(list(instance_discovery.keys()))
            orig_to_serial_ids, serial_to_orig_ids = self.serialize_instance_ids(orig_instance_ids)
            assert orig_instance_ids == list(orig_to_serial_ids.keys())
            metadata["orig_to_serial_ids"] = orig_to_serial_ids
            metadata["serial_to_orig_ids"] = serial_to_orig_ids

            metadata["instance_discovery"] = {orig_to_serial_ids[inst_id]: fr_idx 
                                              for inst_id, fr_idx in instance_discovery.items()}
            
            # extract binary instance masks from semantic map of each frame
            sequence_instances_serial_ids = sorted(list(metadata["instance_discovery"].keys()))
            instance_masks = []
            for fr_idx, fr_map in enumerate(semantic_maps):
                fr_inst_masks = []
                for inst_id in sequence_instances_serial_ids:
                    # semantic map is still labeled with original instance IDs
                    orig_inst_id = serial_to_orig_ids[inst_id]
                    if orig_inst_id in instances_per_frame[fr_idx]:
                        fr_inst_masks.append((fr_map == orig_inst_id).astype('uint8'))
                    else:
                        # if an instance is absent, pad it with empty mask
                        fr_inst_masks.append(np.zeros_like(fr_map).astype('uint8'))
                instance_masks.append(np.stack(fr_inst_masks))
            
            instance_masks = np.stack(instance_masks)    # [T,N,H,W]
            metadata["instance_masks"] = instance_masks

            # recreate semantic maps with serial instance IDs
            semantic_maps_serial = []
            for fr_idx, fr_inst_masks in enumerate(instance_masks):
                fr_map = np.zeros_like(fr_inst_masks[0])
                for inst_id, inst_mask in zip(sequence_instances_serial_ids, fr_inst_masks):
                    fr_map[np.where(inst_mask==1)] = inst_id
                semantic_maps_serial.append(fr_map.astype('uint8'))
                
                # update record on which instances are present in this frame
                instances_per_frame[fr_idx] = [orig_to_serial_ids[orig_inst_id] for orig_inst_id in instances_per_frame[fr_idx]]
            
            metadata["semantic_maps"] = np.stack(semantic_maps_serial)
            metadata["instances_per_frame"] = instances_per_frame

            metadata["clip_length"] = self.clip_length
            metadata["num_overlapping_frames"] = self.num_overlapping_frames

            sequence_annotations.append(metadata)

        return sequence_annotations
    

###########################

# if __name__ == "__main__":
#     # import sys
#     # sys.path.append(os.environ["DYNAMITE_VIDEO_WORKSPACE"])
#     # prepare config variable
#     from detectron2.config import CfgNode as CN
#     cfg = CN()
#     cfg.TRAINING = CN()
#     cfg.TRAINING.CLIP_LENGTH = 4
#     cfg.TRAINING.MIN_MASK_AREA = 400
#     cfg.DATASETS = CN()
#     cfg.DATASETS.VIPSEG = CN()
#     cfg.DATASETS.VIPSEG.TRAINING = CN()
#     cfg.DATASETS.VIPSEG.TRAINING.FPS = 5
#     cfg.DATASETS.VIPSEG.TRAINING.FRAME_SAMPLING_MULTIPLICATIVE_FACTOR = 3.0
    
#     vipseg_loader = VIPSEGTrainingDataset(cfg, 1)
    
#     annotations_content = vipseg_loader.annotations_content