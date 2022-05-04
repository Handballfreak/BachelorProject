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
        v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name, )
    cfg.DATASETS.TEST = (test_dataset_name, )

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025 #0.00025
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


def onImage(imagePath, predictor):
    image = cv2.imread(imagePath)
    outputs = predictor(image)
    v = Visualizer(image[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.figure(figsize=(14, 10))
    plt.imshow(v.get_image())
    plt.show()


def on_video(videoPath, predictor):
    cap = cv2.VideoCapture(videoPath)
    if cap.isOpened()==False:
        print("Error opening file...")
        return
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Result", output.get_image()[:,:,::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()


def register_dataset():
    train_dataset_name = "Dd_train"
    train_images_path = "data/train"
    train_json_annot_path = "data/train/COCO_train_annos.json"  # "data/train/COCO_train_annos.json"

    test_dataset_name = "Dd_val"
    test_images_path = "data/val"
    test_json_annot_path = "data/val/COCO_val_annos.json"  # "data/val/COCO_val_annos.json"

    register_coco_instances(name=train_dataset_name, metadata={}, json_file=train_json_annot_path,
                            image_root=train_images_path)
    register_coco_instances(name=test_dataset_name, metadata={}, json_file=test_json_annot_path,
                            image_root=test_images_path)
