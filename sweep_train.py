from detectron2.utils.logger import setup_logger
from utils import register_dataset
import wandb
import json
import time
import default_config
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from AugTrainer import AugTrainer
import os
import pickle
import numpy as np
from detectron2.data import DatasetCatalog
from utils import plot_samples, get_train_cfg
from evaluate import evaluate
import detectron2
setup_logger()
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


output_dir = "./output/instance_segmentation"
num_classes = 1  # 1
train_dataset_name = "Grey_Thermal_8bit_train"
test_dataset_name = "Grey_Thermal_8bit_val"
device = "cuda"

cfg_save_path = "IS_cfg.pickle"

#####################################################
register_dataset()


# plot_samples(dataset_name=train_dataset_name, n = 2)

########################################################
def main(cfg):
    with open(cfg_save_path, "wb") as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    # trainer = AugTrainer(cfg)
    # wandb.watch(trainer)
    trainer.resume_or_load(resume=False)
    # train model
    # bsp = DatasetCatalog.get("Dd_train")
    # mapper = detectron2.data.DatasetMapper(cfg, is_train=True)
    # bsp2 = detectron2.data.build_detection_train_loader(bsp, mapper=mapper, total_batch_size=1)
    # for i in bsp2:
    #     print(i)

    trainer.train()


def run():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,
                        output_dir)
    # create config informations
    hyperparameters = default_config.config
    wandb.init(project="damage-detection2", entity="handballfreak", config=hyperparameters, reinit=True)
    config = wandb.config
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size
    cfg.SOLVER.BASE_LR = config.learning_rate  # 0.00025
    cfg.SOLVER.MAX_ITER = config.epochs
    cfg.DATALOADER.NUM_WORKERS = config.worker
    cfg.SOLVER.WEIGHT_DECAY = config.weight_decay
    cfg.SOLVER.GAMMA = config.gamma
    cfg.SOLVER.MOMENTUM = config.momentum
    run_name = wandb.run.name
    directory = "./output/instance_segmentation/" + run_name
    cfg.OUTPUT_DIR = directory
    # wandb.tensorboard.patch(root_logdir= directory)
    main(cfg)
    # Training Data log
    with open(directory + "/metrics.json", "r") as metric:
        for line in metric:
            line = json.loads(line)
            wandb.log(line)
    inference = evaluate(test_dataset_name)
    res = {
        "bbox_AP": inference["bbox"]["AP"],
        "bbox_AP50": inference["bbox"]["AP50"],
        "bbox_AP75": inference["bbox"]["AP75"],
        "bbox_APs": inference["bbox"]["APs"],
        "bbox_APm": inference["bbox"]["APm"],
        "bbox_APl": inference["bbox"]["APl"],
        "segm_AP": inference["segm"]["AP"],
        "segm_AP50": inference["segm"]["AP"],
        "segm_AP75": inference["segm"]["AP"],
        "segm_APs": inference["segm"]["AP"],
        "segm_APm": inference["segm"]["AP"],
        "segm_APl": inference["segm"]["AP"],
    }
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
