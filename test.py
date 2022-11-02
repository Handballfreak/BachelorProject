from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *

cfg_save_path = "IS_cfg.pickle"
register_datasets()

with open(cfg_save_path, "rb") as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

# Inserting the path for image and video to test the model
image_path = ""
videoPath = ""

# remove # in Line 23 for plot of n samples from choosen dataset
# (RGB_train, RGB_val, RGB_Thermal_train, RGB_Thermal_val, Grey_Thermal_train, Grey_Thermal_val)
# plot_samples(dataset_name="", n = 4)
on_image(image_path, predictor)
# Line below remove # for video test
# on_video(videoPath, predictor)
