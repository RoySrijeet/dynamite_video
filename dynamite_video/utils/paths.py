import os

class Paths:
    """
    A class to manage paths to relevant directories
    """

    def __init__(self):
        raise ValueError("Static class 'Paths' should be not be initialized")
    
    @classmethod
    def to_workspace(cls):
        """
        Path to the directory containing `configs`, `datasets`, `checkpoints`, 
        `dynamite_video` etc.
        """
        return os.environ["DYNAMITE_VIDEO_WORKSPACE"]

    @classmethod
    def to_configs(cls):
        """Path to all configuration files"""
        return os.path.join(cls.to_workspace(), "configs")
    
    @classmethod
    def to_pretrained_weights(cls):
        """Path to all pretrained weights (for backbones etc.)"""
        return os.path.join(cls.to_workspace(), "pretrained_weights")
    
    @classmethod
    def to_datasets_root(cls):
        """Path to datasets root directory"""
        return os.path.join(cls.to_workspace(), "dataset_juwels")  # TODO
    
    @classmethod
    def to_training_images(cls):
        """Path to base directory containing training images"""
        return os.path.join(cls.to_datasets_root(), "images/training")

    @classmethod
    def to_evaluation_images(cls):
        """Path to base directory containing evaluation images"""
        return os.path.join(cls.to_datasets_root(), "images/evaluation")

    @classmethod
    def to_training_annotations(cls):
        """Path to base directory containing training annotations"""
        return os.path.join(cls.to_datasets_root(), "annotations/training")
    
    @classmethod
    def to_evaluation_annotations(cls):
        """Path to base directory containing evaluation annotations"""
        return os.path.join(cls.to_datasets_root(), "annotations/evaluation")

    # --------------------- DAVIS --------------------- #

    @classmethod
    def to_davis_training_images(cls):
        """Path to DAVIS-2017 JPEGImages"""
        return os.path.join(cls.to_training_images(), "davis")
    
    @classmethod
    def to_davis_training_annotations(cls):
        """Path to DAVIS-2017 Training Annotations JSON file"""
        return os.path.join(cls.to_training_annotations(), "davis_semisupervised.json")

    # @classmethod
    # def to_davis_train_imset(cls):
    #     """Path to DAVIS-2017 training set list file"""
    #     return os.path.join(cls.to_davis_root(), "ImageSets/2017/train.txt")
    
    # @classmethod
    # def to_davis_val_imset(cls):
    #     """Path to DAVIS-2017 training set list file"""
    #     return os.path.join(cls.to_davis_root(), "ImageSets/2017/val.txt")


    # --------------------- MOSE --------------------- #

    # @classmethod
    # def to_mose_root(cls):
    #     """Path to MOSE root directory"""
    #     return os.path.join(cls.to_datasets_root(), "MOSE")
    
    # @classmethod
    # def to_mose_train_images(cls):
    #     """Path to MOSE training set images"""
    #     return os.path.join(cls.to_mose_root(), "train/JPEGImages")
    
    # @classmethod
    # def to_mose_train_annotations(cls):
    #     """Path to MOSE training set images"""
    #     return os.path.join(cls.to_mose_root(), "train/Annotations")
    
    # @classmethod
    # def to_mose_train_imset(cls):
    #     """Path to MOSE training split sequences""" # NOTE: this is custom subset
    #     return os.path.join(cls.to_mose_root(), "train/MOSE_sample_train_list.txt")
    
    # @classmethod
    # def to_mose_val_images(cls):
    #     """Path to MOSE validation set images"""
    #     return os.path.join(cls.to_mose_root(), "valid/JPEGImages")
    
    # @classmethod
    # def to_mose_val_annotations(cls):
    #     """Path to MOSE validation set images"""
    #     return os.path.join(cls.to_mose_root(), "valid/Annotations")
    
    # @classmethod
    # def to_mose_val_imset(cls):
    #     """Path to MOSE val split sequences""" # NOTE: this is custom subset
    #     return os.path.join(cls.to_mose_root(), "val/MOSE_sample_val_list.txt")
    

    # --------------------- PUMaVOS --------------------- #

    # @classmethod
    # def to_pumavos_root(cls):
    #     """Path to PUMaVOS root directory"""
    #     return os.path.join(cls.to_datasets_root(), "PUBLIC_PUMaVOS")

    # @classmethod
    # def to_pumavos_images(cls):
    #     """Path to PUMaVOS JPEGImages"""
    #     return os.path.join(cls.to_pumavos_root(), "JPEGImages")
    
    # @classmethod
    # def to_pumavos_annotations(cls):
    #     """Path to PUMaVOS Annotations"""
    #     return os.path.join(cls.to_pumavos_root(), "Annotations")
    
    # @classmethod
    # def to_pumavos_imset(cls):
    #     """Path to PUMaVOS video list file"""
    #     return os.path.join(cls.to_pumavos_root(), "imset.txt")
    

    # --------------------- BURST --------------------- #

    @classmethod
    def to_burst_training_images(cls):
        """Path to BURST training images (all_frames)"""
        return os.path.join(cls.to_training_images(), "burst")
    
    @classmethod
    def to_burst_training_annotations(cls):
        """Path to BURST training annotation JSON file"""
        return os.path.join(cls.to_training_annotations(), "burst.json")
    
    
    @classmethod
    def to_burst_val_images(cls):
        """Path to BURST validation images (all_frames)"""
        return os.path.join(cls.to_burst_root(), "all_frames/val")
    
    # @classmethod
    # def to_burst_val_annotations(cls):
    #     """Path to BURST validation annotation JSON file"""
    #     return {
    #         "all_classes": os.path.join(cls.to_burst_root(), "annotations/val/all_classes.json"),
    #         "first_frame_annotations": os.path.join(cls.to_burst_root(), "annotations/val/first_frame_annotations.json"),
    #     }
    
    # @classmethod
    # def to_burst_test_images(cls):
    #     """Path to BURST test images (all_frames)"""
    #     return os.path.join(cls.to_burst_root(), "all_frames/test")
    
    # @classmethod
    # def to_burst_val_annotations(cls):
    #     """Path to BURST test annotation JSON file"""
    #     return {
    #         "all_classes": os.path.join(cls.to_burst_root(), "annotations/test/all_classes.json"),
    #         "first_frame_annotations": os.path.join(cls.to_burst_root(), "annotations/test/first_frame_annotations.json"),
    #     }
    
    # --------------------- KITTI-STEP --------------------- #

    @classmethod
    def to_kitti_step_training_images(cls):
        """Path to KITTI-STEP training images"""
        return os.path.join(cls.to_training_images(), "kitti_step")

    @classmethod
    def to_kitti_step_train_annotations(cls):
        """Path to KITTI-STEP training annotation JSON"""
        return os.path.join(cls.to_training_annotations(), "kitti_step_train.json")
    
    @classmethod
    def to_kitti_step_trainval_annotations(cls):
        """Path to KITTI-STEP train-val annotation JSON"""
        return os.path.join(cls.to_training_annotations(), "kitti_step_trainval.json")
    
    # @classmethod
    # def to_kitti_step_trainval_images(cls):
    #     """Path to KITTI-STEP training images"""
    #     return os.path.join(cls.to_kitti_step_root(), "KITTI-STEP/data_tracking_image_2/training/image_02")

    # @classmethod
    # def to_kitti_step_test_images(cls):
    #     """Path to KITTI-STEP test images"""
    #     return os.path.join(cls.to_kitti_step_root(), "KITTI-STEP/data_tracking_image_2/testing/image_02")
    

    # ------------------------ VIPSeg ----------------------- #

    @classmethod
    def to_vipseg_training_images(cls):
        """Path to VIPSeg images"""
        return os.path.join(cls.to_training_images(), "vipseg")
    
    @classmethod
    def to_vipseg_annotations(cls):
        """Path to VIPSeg annotations"""
        return os.path.join(cls.to_training_annotations(), "vipseg/panoptic_masks")
    
    @classmethod
    def to_vipseg_train_video_info(cls):
        """Path to VIPSeg training set list file"""
        return os.path.join(cls.to_training_annotations(), "vipseg/video_info.json")
    
    # @classmethod
    # def to_vipseg_train_imset(cls):
    #     """Path to VIPSeg training set list file"""
    #     return os.path.join(cls.to_vipseg_root(), "train.txt")
    
    # @classmethod
    # def to_vipseg_val_imset(cls):
    #     """Path to VIPSeg training set list file"""
    #     return os.path.join(cls.to_vipseg_root(), "val.txt")
    
    # @classmethod
    # def to_vipseg_test_imset(cls):
    #     """Path to VIPSeg training set list file"""
    #     return os.path.join(cls.to_vipseg_root(), "test.txt")
    
    
    # ------------------- Cityscapes-VPS ------------------- #

    # @classmethod
    # def to_cityscapes_vps_root(cls):
    #     """Path to Cityscapes VPS root directory"""
    #     return os.path.join(cls.to_datasets_root(), "CITYSCAPES-VPS")
    
    # @classmethod
    # def to_cityscapes_vps_train_images(cls):
    #     """Path to Cityscapes VPS training images"""
    #     return os.path.join(cls.to_cityscapes_vps_root(), "cityscapes_vps/train/img")
    
    # @classmethod
    # def to_cityscapes_vps_train_annotations(cls):
    #     """Path to Cityscapes VPS training JSON annotation file"""
    #     return os.path.join(cls.to_cityscapes_vps_root(), "json_anno/train/cityscapes_vps.json")

    # @classmethod
    # def to_cityscapes_vps_val_images(cls):
    #     """Path to Cityscapes VPS val images"""
    #     return os.path.join(cls.to_cityscapes_vps_root(), "cityscapes_vps/val/img_all")
    
    # @classmethod
    # def to_cityscapes_vps_val_annotations(cls):
    #     """Path to Cityscapes VPS val split JSON annotation file"""
    #     return os.path.join(cls.to_cityscapes_vps_root(), "json_anno/val/im_all_info_val_city_vps.json")
    
    
    # --------------------- YouTube-VOS --------------------- #

    # --------------------- YouTube-VIS --------------------- #