import cv2
import imgaug
import imgaug.augmenters as iaa
import json
import numpy as np
import os
import sys
sys.path.append(os.environ["DYNAMITE_VIDEO_WORKSPACE"])
import random
import torch

from torch.utils.data import Dataset

from dynamite_video.data.utils.clicker import get_clicks_coords
from dynamite_video.data.utils.data_utils import serialize_object_ids
from dynamite_video.data.utils.file_packer import FilePackReader
from dynamite_video.utils.paths import Paths


class PseudoVideoTrainingDataset(Dataset):
    """
    Pseudo Video Training Dataset Class

    Creates a `torch.utils.data.Dataset` class to load pseudo-videos from COCO
    """
    
    def __init__(
            self, 
            cfg, 
            dataset_name: str, 
            path_to_images: str,
            path_to_annotations: str,
            path_to_json_annotations: str,
            path_to_categories_info: str,
            num_samples: int,
    ):
        """
        Args:
            cfg: detectron2-style configuration
            dataset_name: str, name of the image dataset
            path_to_images: str, path to dataset images
            path_to_annotations: str, path to annotation files
            path_to_json_annotations: str, path to JSON file with annotation info
            path_to_categories_info: str, path to JSON file with dataset category info
            num_samples: int, number of training samples from the dataset
        """
        
        self.cfg = cfg
        self.name = dataset_name
        self.clip_length = cfg.TRAINING.CLIP_LENGTH
        self.num_samples = num_samples

        # path to training images
        self.path_to_images = path_to_images
        if not os.path.exists(self.path_to_images):
            # if path does not exist, perhaps we're on JUWELS
            path_to_images = f"{Paths.to_training_images_on_juwels()}/{dataset_name}.fpack"
            assert os.path.exists(path_to_images), f"{dataset_name} images not found at: {self.path_to_images} or {path_to_images}"
            self.path_to_images = path_to_images
            self.fpack_reader = FilePackReader(self.path_to_images, multiprocess_lock=False)

        # path to panoptic maps
        self.path_to_annotations = path_to_annotations
        assert os.path.exists(path_to_annotations), f"{dataset_name} masks not found at: {self.path_to_annotations}"
        
        # read category information
        self.parse_json_category(path_to_categories_info)

        # read annotations from json
        parsed_annotations = self.parse_json_annotations(path_to_json_annotations)
        # filter out zero instance images
        self.image_samples = self.filter_zero_instance_images(parsed_annotations)
        self.image_ids = list(self.image_samples.keys())
        self.fallback_candidates = set(self.image_ids)

        # output size
        self.output_dims = cfg.INPUT.AUGMENTATION.IMAGE_SIZE
        
        # color augmentations
        # NOTE: imgaug acts freaky with multiple workers (due to global cache _LUT_CACHE); set cfg.DATALOADER.NUM_WORKERS = 0
        self.color_augmenter = iaa.Sequential([
            # color tone and vividness variation simulating white balance settings           
            iaa.AddToHueAndSaturation(value_hue=(-12, 12), value_saturation=(-12, 12)),
            # slightly dampen/pronoune contrast to lower sensitivity to lighting differences
            iaa.LinearContrast(alpha=(0.95, 1.05)),
            # small brightness variations to simulate different exposures and illumination conditions
            iaa.AddToBrightness(add=(-25, 25))
        ])

        # geometric transformations
        # apply affine transformations; uses default behaviour and assigns 0 to any "new" pixels created
        self.deterministic = cfg.TRAINING.PRETRAIN_DETERMINISTIC_AUG
        if self.deterministic:
            self.augmentation_affine = iaa.Affine(scale=(0.9, 1.1), rotate=(-8, 8), shear=(-5, 5))
        else:
            self.augmentation_affine = iaa.Affine(scale=(0.9, 1.1), rotate=(-20, 20), shear=(-10, 10))
        self.augmentation_fixed_crop = iaa.CropToFixedSize(self.output_dims[0], self.output_dims[1])
        self.augmentation_min_dims = [self.output_dims[0], self.output_dims[0] + 32, self.output_dims[0] + 64]


    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        index = self.image_ids[index % len(self.image_ids)]
        n_tries = 0
        while True:
            try:
                sample = self.parse_image(index)
                return sample

            except:
                self.fallback_candidates.discard(index)
                index = random.choice(list(self.fallback_candidates))
                n_tries += 1
                if n_tries % 3 == 0:
                    print(f"Num failed tries: {n_tries} for dataset {self.name}")


    def parse_image(self, index: str):
        image_struct = self.image_samples[index]

        # read image file
        image = cv2.imread(os.path.join(self.path_to_images, image_struct["file_name"]), cv2.IMREAD_COLOR)  # [H, W, 3] (BGR)
        if self.cfg.INPUT.RGB:
            # BGR -> RGB
            image = image[:, :, ::-1]
        # apply color augmentation
        image = self.color_augmenter(image=image)

        # read PNG mask
        png_mask = cv2.imread(os.path.join(self.path_to_annotations, image_struct["file_name_png"]), cv2.IMREAD_COLOR)  # [H, W, 3] (BGR)
        # convert mask to panoptic map with target IDs as labels
        png_mask = png_mask.astype(np.int64)
        png_mask = (256 ** 2 * png_mask[:, :, 0]) + (256 * png_mask[:, :, 1]) + png_mask[:, :, 2]   # [H, W]

        # apply random horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, 1)
            png_mask = np.flip(png_mask, 1)

        # generate pseudo-video from current image
        images, orig_panoptic_masks = self.apply_geometric_augmentations(image, png_mask)
        images = np.transpose(images, (0, 3, 1, 2))   # [T, H, W, 3] -> [T, 3, H, W]

        # IDs of target object in the pseudo-video
        orig_object_ids = list(np.unique(orig_panoptic_masks))
        orig_object_ids = [obj_id for obj_id in orig_object_ids if obj_id not in self.ignore_classes]
        assert len(orig_object_ids) >= 1, f"no objects in the sampled clip."
        # serialize object ids
        orig_to_serial_id, serial_to_orig_id = serialize_object_ids(orig_object_ids)
        
        # serialize object ids in panoptic map and obtain binary masks
        panoptic_masks = np.zeros_like(orig_panoptic_masks, dtype=np.uint8)
        T,H,W = panoptic_masks.shape
        N = len(orig_object_ids)
        binary_masks = np.zeros((T,N,H,W), dtype=np.uint8)
        for orig_id, serial_id in orig_to_serial_id.items():
            panoptic_masks[np.where(orig_panoptic_masks==orig_id)] = serial_id
            binary_masks[:, serial_id-1] = (panoptic_masks==serial_id).astype(np.uint8)
        
        # consider all parts of the image without fg label as bg
        bg_masks = (panoptic_masks==0)  # [T, H, W]
                
        # record the category IDs of the objects in the clip
        target_categories = dict()
        for ann_segment in image_struct["segments"]:
            if ann_segment["id"] not in orig_object_ids:
                continue
            target_categories[orig_to_serial_id[ann_segment["id"]]] = ann_segment["category_id"]
        
        # sample clicks
        (num_clicks_per_target, 
        fg_coords_list, 
        bg_coords_list, 
        max_timestamp_list) = get_clicks_coords(target_masks=binary_masks, 
                                                max_num_points=self.cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE,
                                                optional_frames_fg_prob=self.cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB,
                                                bg_masks=bg_masks,
                                                bg_prob=self.cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB,
                                                gamma=self.cfg.CLICKER.TRAINING.GAMMA,
                                                start_t=1,
                                            )
        
        if not all(np.sum(num_clicks_per_target, axis=0)):
            raise "One or more targets in this clip did not receive a click!"
        
        meta_info = {
            "orig_dims": images.shape[2:],
            "seq_name": f"{self.name}_{index}",
            "frame_indices": [i for i in range(self.clip_length)],
            "orig_to_serial_id": orig_to_serial_id,
            "serial_to_orig_id": serial_to_orig_id, 
            "ignore_class": self.ignore_classes,
            "target_categories": target_categories,
            "max_instances_per_category": 0,
        }

        return {
            # T,3,H,W image tensors, not normalized, padded region has value 128
            "images": torch.as_tensor(images, dtype=torch.uint8),
            
            # T,N,H,W binary masks of target objects. This includes the semantic maps of the `stuff` classes 
            # and the instance maps of `thing` class instances. Which means, the binary masks do not include 
            # region of `thing` class that is not covered by corresponding instances, and the VOID class.
            "binary_masks": torch.as_tensor(binary_masks, dtype=torch.uint8),
            
            # H,W boolean padding mask where padded region is labeled True
            "padding_mask": torch.zeros((H,W), dtype=torch.bool),
            
            # T,H,W boolean mask for `VOID` class
            "ignore_masks": torch.zeros((T,H,W), dtype=torch.bool),
            
            # T,H,W bg mask includes any region not covered by the target object binary masks
            # NOTE: this includes ignore mask regions as well
            "bg_masks": torch.as_tensor(bg_masks, dtype=torch.uint8),
            
            # T,H,W semantic map with serialized target IDs. Background gets labeled 0
            "panoptic_masks": torch.as_tensor(panoptic_masks, dtype=torch.uint8),

            # T,N array recording num clicks on each target in each frame
            "num_clicks_per_object": num_clicks_per_target,
            
            # list of fg clicks sampled on the clip. Each click follows the format: [y,x,i,f,t]
            "fg_coords_list": fg_coords_list,
            
            # list of bg clicks sampled on the clip. Each click follows the format: [y,x,-1,f,t]
            "bg_coords_list": bg_coords_list,

            # list of length T recording timestamp of the latest click on each frame
            "max_timestamp_list": max_timestamp_list,

            # info about the clip and its source video
            "meta": meta_info,
        }
    

    def apply_geometric_augmentations(self, image: np.ndarray, panoptic_mask: np.ndarray):
        """
        Apply geomtric transformations (affine), resize, and crop to target resolution.
        Augmented frames constitute the consecutive frames of the pseudo-video.
        
        Args:
            image: np.ndarray, shape [H, W, 3]
            panoptic_mask: np.ndarray, shape [H, W]
        """
        # deterministic augmentation applies the same affine transformations on successive frames
        min_dim = None
        if self.deterministic:
            self.augmentation_affine.to_deterministic()
            self.augmentation_fixed_crop.to_deterministic()
            min_dim = self.augmentation_min_dims[torch.randint(len(self.augmentation_min_dims), (1,)).item()]
            

        # prepare imgaug segmentation object; treats its values as categorical labels, 
        # instead of continuous pixel intensities when applying interpolation (NN)
        seg = imgaug.SegmentationMapsOnImage(panoptic_mask.astype(np.int32), shape=image.shape)

        seq_images, seq_panoptic_masks = [], []

        for _ in range(self.clip_length):
            # affine augmentation
            _image, _seg = self.augmentation_affine(image=image, segmentation_maps=seg)
            _mask = _seg.get_arr()  # [H, W], int32

            # resize
            _image, _mask = self.random_resize(_image, _mask, min_dim=min_dim)
            
            # crop
            _seg = imgaug.SegmentationMapsOnImage(_mask, shape=_image.shape)
            _image, _seg = self.augmentation_fixed_crop(image=_image, segmentation_maps=_seg)
            _mask = _seg.get_arr()  # [H, W]

            seq_images.append(_image)
            seq_panoptic_masks.append(_mask)

            if self.deterministic:
                image, seg = _image, _seg

        # stack over time
        seq_images = np.stack(seq_images, axis=0)              # [T, H, W, 3]
        seq_panoptic_masks = np.stack(seq_panoptic_masks, 0)   # [T, H, W]
        return seq_images, seq_panoptic_masks


    def random_resize(self, image: np.ndarray, masks: np.ndarray, min_dim: int=None):
            """
            Resize while preserving aspect ratio

            Args:
                image: np.ndarray, shape [H, W, 3]
                masks: np.ndarray, shape [H, W]
            """
            height, width = image.shape[:2]
            dims = [height, width]
            lower_size = float(min(dims))
            higher_size = float(max(dims))

            if min_dim is None:
                min_dim = self.augmentation_min_dims[torch.randint(len(self.augmentation_min_dims), (1,)).item()]
            scale_factor = min_dim / lower_size
            # if (higher_size * scale_factor) > 1333:
            #     scale_factor = 1333 / higher_size

            new_height, new_width = round(scale_factor * height), round(scale_factor * width)

            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR_EXACT)
            masks = cv2.resize(masks, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT) 
            return image, masks
    

    def parse_json_annotations(self, filepath: str):
        """
        Reads annotation metadata from a JSON file
        """
        with open(filepath, 'r') as fh:
            content = json.load(fh)

        images_struct = dict()
        # read image metadata
        for img in content["images"]:
            images_struct[img['id']] = {
                "file_name": img["file_name"],
                "height": img['height'],
                "width": img['width'],
                "segments": []
            }
        # instance and annotation info
        for ann in content["annotations"]:
            images_struct[ann['image_id']]["segments"] = ann["segments_info"]
            images_struct[ann['image_id']]["file_name_png"] = ann["file_name"]

        return images_struct


    def filter_zero_instance_images(self, images_struct):
        """
        Filter out images with no instances ("thing")
        """
        image_ids = list(images_struct.keys())
        n_removed = 0
            
        for img_id, entry in images_struct.items():
            instance_found = False
            for ann_segment in entry["segments"]:
                cat_id = ann_segment["category_id"]
                if cat_id not in self.stuff_classes:
                    instance_found = True
                    break

            if not instance_found:
                n_removed += 1
                image_ids.remove(img_id)

        parsed_annotations = {
            img_id: images_struct[img_id]
            for img_id in image_ids
        }
        return parsed_annotations
    

    def parse_json_category(self, path_to_json: str):
        """
        Given path to a JSON file with dataset category info, read:
            * categories: dict with ID-name mapping
            * isthing: list of thing class IDs
            * isstuff: list of stuff class IDs
        """
        with open(path_to_json, "r") as f:
            category_info = json.load(f)
        
        self.categories = {}
        self.thing_classes = []
        self.stuff_classes = []
        for entry in category_info:
            self.categories[entry["id"]] = entry["name"]
            if entry["isthing"]:
                self.thing_classes.append(entry["id"])
            else:
                self.stuff_classes.append(entry["id"])


    @property
    def ignore_classes(self):
        return [0]


class COCOPanopticDataset(PseudoVideoTrainingDataset):
    def __init__(self, cfg, num_samples):
        
        super().__init__(
            cfg=cfg,
            dataset_name="coco",
            path_to_images=Paths.to_coco_images(),
            path_to_annotations=Paths.to_coco_annotations(),
            path_to_json_annotations=Paths.to_coco_json_annotations(),
            path_to_categories_info=Paths.to_coco_category_info(),
            num_samples=num_samples,
        )


class ADE20KPanopticDataset(PseudoVideoTrainingDataset):
    def __init__(self, cfg, num_samples):
        
        super().__init__(
            cfg=cfg,
            dataset_name="ade20k",
            path_to_images=Paths.to_ade20k_images(),
            path_to_annotations=Paths.to_ade20k_annotations(),
            path_to_json_annotations=Paths.to_ade20k_json_annotations(),
            path_to_categories_info=Paths.to_ade20k_category_info(),
            num_samples=num_samples,
        )



###########################

# if __name__ == "__main__":
#     # add these import lines
#     # import sys
#     # sys.path.append(os.environ["DYNAMITE_VIDEO_WORKSPACE"])
    
#     # prepare config variable
#     from torch.utils.data import DataLoader
#     from detectron2.config import CfgNode as CN
#     from dynamite_video.data.utils.collate import collate_fn_pretrain
#     cfg = CN()
#     cfg.INPUT = CN()
#     cfg.INPUT.RGB = True
#     cfg.INPUT.AUGMENTATION = CN()
#     cfg.INPUT.AUGMENTATION.IMAGE_SIZE = [512,512]
#     cfg.CLICKER = CN()
#     cfg.CLICKER.TRAINING = CN()
#     cfg.CLICKER.TRAINING.MAX_NUM_CLICKS_PER_INSTANCE = 6
#     cfg.CLICKER.TRAINING.OPTIONAL_FRAMES_FG_SAMPLE_PROB = 1.
#     cfg.CLICKER.TRAINING.BACKGROUND_SAMPLING_PROB = 0.
#     cfg.CLICKER.TRAINING.GAMMA = 0.7
#     cfg.TRAINING = CN()
#     cfg.TRAINING.CLIP_LENGTH = 4
#     cfg.TRAINING.PRETRAIN_DETERMINISTIC_AUG = True
    
#     coco_dataset = COCOPanopticDataset(cfg, 20000)
#     data_loader = DataLoader(coco_dataset, batch_size=4, num_workers=4, collate_fn=collate_fn_pretrain)
#     data_loader = iter(data_loader)
#     i = 0
#     while True:
#         training_sample = next(data_loader)
#         if i%100 == 0:
#             print(i)
#         i += 1
#         if i == 19999:
#             break