from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

import random
import cv2
import matplotlib.pyplot as plt


def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,
                  output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025  # 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.STEPS = []
    cfg.INPUT.FORMAT = "BGR"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg


def on_image(image_path, predictor):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image())
    plt.show()


def on_video(video_path, predictor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening file...")
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()


def register_dataset():
    register_grey_thermal_8bit_dataset()
    register_rgb_dataset()
    register_rgb_thermal_dataset()

def register_grey_thermal_8bit_dataset():
    train_dataset_name = "Grey_Thermal_8bit_train"
    train_images_path = "data/FLIR_ADAS_1_3_neu/images_thermal_train"
    train_json_annot_path = "data/FLIR_ADAS_1_3/train/annotations_allgemein.json"
    train_json_annot_path = "data/FLIR_ADAS_1_3_neu/images_thermal_train/filtered_coco.json"

    val_dataset_name = "Grey_Thermal_8bit_val"
    val_images_path = "data/FLIR_ADAS_1_3_neu/images_thermal_val/"
    val_json_annot_path = "data/FLIR_ADAS_1_3/val/annotations_allgmein.json"
    val_json_annot_path = "data/FLIR_ADAS_1_3_neu/images_thermal_val/filtered_coco.json"

    register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path,
                            image_root=train_images_path)
    register_coco_instances(name=val_dataset_name, metadata={}, json_file=val_json_annot_path,
                            image_root=val_images_path)


def register_rgb_dataset():
    train_dataset_name = "RGB_train"
    train_images_path = "data/FLIR_ADAS_1_3_neu/images_rgb_train"
    train_json_annot_path = "data/FLIR_ADAS_1_3/train/annotations_allgemein.json"
    train_json_annot_path = "data/FLIR_ADAS_1_3_neu/images_rgb_train/filtered_coco.json"

    val_dataset_name = "RGB_val"
    val_images_path = "data/FLIR_ADAS_1_3_neu/images_rgb_val"
    val_json_annot_path = "data/FLIR_ADAS_1_3/val/annotations_allgmein.json"
    val_json_annot_path = "data/FLIR_ADAS_1_3_neu/images_rgb_val/filtered_coco.json"

    register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path,
                            image_root=train_images_path)
    register_coco_instances(name=val_dataset_name, metadata={}, json_file=val_json_annot_path,
                            image_root=val_images_path)


def register_rgb_thermal_dataset():
    train_dataset_name = "RGB_Thermal_8bit_train"
    train_images_path = "data/ALL_IN_ONE_RGB_IMG_ANOT/train/"
    train_json_annot_path = "data/ALL_IN_ONE_RGB_IMG_ANOT/train\\train_mod_coco.json"

    val_dataset_name = "RGB_Thermal_8bit_val"
    val_images_path = "data/ALL_IN_ONE_RGB_IMG_ANOT/test/"
    val_json_annot_path = "data/ALL_IN_ONE_RGB_IMG_ANOT/test\\test_mod_coco.json"

    register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path,
                            image_root=train_images_path)
    register_coco_instances(name=val_dataset_name, metadata={}, json_file=val_json_annot_path,
                            image_root=val_images_path)
