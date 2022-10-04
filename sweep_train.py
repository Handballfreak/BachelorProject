from detectron2.utils.logger import setup_logger
from utils import register_datasets
import wandb
import json
import time
import default_config
from detectron2.engine import DefaultTrainer
from AugTrainer import AugTrainer
import os
import pickle
from utils import get_train_cfg
from evaluate import evaluate


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

# Datasets available
# train_dataset_name = "RGB_train"
# test_dataset_name = "RGB_val"
train_dataset_name = "RGB_Thermal_train"
test_dataset_name = "RGB_Thermal_val"
# train_dataset_name = "Grey_Thermal_train"
# test_dataset_name = "Grey_Thermal_val"

########################################################
register_datasets()
########################################################
def main(cfg):
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

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
    # create config informations
    hyperparameters = default_config.config
    wandb.init(entity="handballfreak", config=hyperparameters, reinit=True)
    config = wandb.config
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    cfg.SOLVER.BASE_LR = config.learning_rate
    cfg.SOLVER.MAX_ITER = config.epochs
    cfg.DATALOADER.NUM_WORKERS = config.worker
    cfg.SOLVER.WEIGHT_DECAY = config.weight_decay
    cfg.SOLVER.GAMMA = config.gamma
    cfg.SOLVER.MOMENTUM = config.momentum
    run_name = wandb.run.name
    directory = "./output/instance_segmentation/" + run_name
    cfg.OUTPUT_DIR = directory
    if wandb.config.Transform == "On":
        main_transform(cfg)
    else:
        main(cfg)
    # Training Data log save
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
    # Artifact bei Sweep raus da es viel Speicherplatz verbraucht
    # artifact = wandb.Artifact("Model_Parameters", "model")
    # artifact.add_file(directory + "/model_final.pth")
    # wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    wandb.tensorboard.patch(root_logdir="./output/instance_segmentation")
    run()
