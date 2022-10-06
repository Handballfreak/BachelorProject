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

image_path = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT\\test\\002.jpg"
videoPath = ""

# plot_samples(dataset_name="", n = 4)
on_image(image_path, predictor)
# Zeile drunter Kommentar Zeichen entfernen für Video Test
#on_video(videoPath, predictor)