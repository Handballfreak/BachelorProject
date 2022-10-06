from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os
import pickle
from utils import register_datasets


def evaluate(test_dataset_name):
    # register_datasets()

    cfg_save_path = "IS_cfg.pickle"

    with open(cfg_save_path, "rb") as f:
        cfg = pickle.load(f)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(test_dataset_name, ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, test_dataset_name)

    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    return inference


if __name__ == "__main__":
    evaluate()
