from detectron2.utils.logger import setup_logger
from utils import register_datasets
import wandb
import json
import time
from detectron2.engine import DefaultTrainer
from AugTrainer import AugTrainer
import os
import pickle
from detectron2.data import DatasetCatalog
from utils import get_train_cfg
from evaluate import evaluate

import default_config

setup_logger()

# Mask R-CNN Model / Instance Segmentation
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# Faster R-CNN Model/ Object Detection
# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


output_dir = "./output/instance_segmentation"
num_classes = 1
device = "cuda"
cfg_save_path = "IS_cfg.pickle"

# available Datasets
# train_dataset_name = "RGB_train"
# test_dataset_name = "RGB_val"
# train_dataset_name = "RGB_Thermal_train"
# test_dataset_name = "RGB_Thermal_val"
train_dataset_name = "Grey_Thermal_train"
test_dataset_name = "Grey_Thermal_val"


########################################################
register_datasets()
########################################################
def main(cfg):
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Inspect Dataset
    # bsp = DatasetCatalog.get("Dd_train")
    # mapper = detectron2.data.DatasetMapper(cfg, is_train=True)
    # bsp2 = detectron2.data.build_detection_train_loader(bsp, mapper=mapper, total_batch_size=1)


    # train model
    trainer.train()


def main_transform(cfg):
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = AugTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()


def run():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,
                        output_dir)
    config = default_config.config
    config["learning_rate"] = 0.004666
    config["batch_size"] = 4
    config["momentum"] = 0.8914
    config["epochs"] = 1500
    config["workers"] = 8
    config["transform"] = "Off"
    config["weight_decay"] = 0.0001
    config["gamma"] = 0.1
    cfg.SOLVER.IMS_PER_BATCH = config["batch_size"]
    cfg.SOLVER.BASE_LR = config["learning_rate"]
    cfg.SOLVER.MAX_ITER = config["epochs"]
    cfg.DATALOADER.NUM_WORKERS = config["workers"]
    cfg.SOLVER.MOMENTUM = config["momentum"]
    cfg.SOLVER.WEIGHT_DECAY = config["weight_decay"]
    cfg.SOLVER.GAMMA = config["gamma"]

    # create config informations
    wandb.init(project="Testproject", entity="handballfreak", config=config, reinit=True)
    run_name = wandb.run.name
    directory = output_dir + "/" + run_name
    cfg.OUTPUT_DIR = directory
    if config["transform"] == "On":
        main_transform(cfg)
    else:
        main(cfg)

    # Training Data log
    with open(directory + "/metrics.json", "r") as metric:
        for line in metric:
            line = json.loads(line)
            wandb.log(line)
    inference = evaluate(test_dataset_name)
    # save results to wandb
    wandb.run.summary["bbox_AP"] = inference["bbox"]["AP"]
    wandb.run.summary["bbox_AP50"] = inference["bbox"]["AP50"]
    wandb.run.summary["bbox_AP75"] = inference["bbox"]["AP75"]
    wandb.run.summary["bbox_APs"] = inference["bbox"]["APs"]
    wandb.run.summary["bbox_APm"] = inference["bbox"]["APm"]
    wandb.run.summary["bbox_APl"] = inference["bbox"]["APl"]
    wandb.run.summary["segm_AP"] = inference["segm"]["AP"]
    wandb.run.summary["segm_AP50"] = inference["segm"]["AP50"]
    wandb.run.summary["segm_AP75"] = inference["segm"]["AP75"]
    wandb.run.summary["segm_APs"] = inference["segm"]["APs"]
    wandb.run.summary["segm_APm"] = inference["segm"]["APm"]
    wandb.run.summary["segm_APl"] = inference["segm"]["APl"]

    # Model Artifact
    time.sleep(2)
    artifact = wandb.Artifact("Model_Parameters", "model")
    artifact.add_file(directory + "/model_final.pth")
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    wandb.tensorboard.patch(root_logdir="./output/instance_segmentation")
    run()
