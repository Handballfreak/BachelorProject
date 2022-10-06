import shutil
import os
import json

# paths from annot file and source of the images
path_files = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU"
path_anot_train_file = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU\Anotations\ALL_IN_ONE_RGB_ANOT_COCO\\train\\train_coco.json"
path_train_dest = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU\Anotations\ALL_IN_ONE_RGB_ANOT_COCO\\train"
path_anot_test_file = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU\Anotations\ALL_IN_ONE_RGB_ANOT_COCO\\test\\test_coco.json"
path_test_dest = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU\Anotations\ALL_IN_ONE_RGB_ANOT_COCO\\test"


def move_train_test_files(path_annot_file, path_destination):
    missing_files = []
    # read annot file
    with open(path_annot_file, "r") as f:
        annot = json.load(f)
    images = annot["images"]
    for image in images:
        file_name = image["file_name"]
        file_path = os.path.join(path_files, file_name)
        file_dest_path = os.path.join(path_destination, file_name)
        # Bild verschieben
        try:
            shutil.move(file_path, file_dest_path)
        except FileNotFoundError:
            missing_files.append(file_name)
    print(path_destination)
    print(missing_files)


move_train_test_files(path_anot_train_file, path_train_dest)
move_train_test_files(path_anot_test_file, path_test_dest)
