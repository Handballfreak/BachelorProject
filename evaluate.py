from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
import os
import pickle
from utils import register_dataset


def evaluate():
    # register_dataset()

    cfg_save_path = "IS_cfg.pickle"

    with open(cfg_save_path, "rb") as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)
    # trainer = DefaultTrainer(cfg)

    evaluator = COCOEvaluator("Dd_val", ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "Dd_val")

    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    # print(inference["bbox"])
    return inference


if __name__ == "__main__":
    evaluate()
