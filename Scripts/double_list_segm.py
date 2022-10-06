import json

file_test = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU\Anotations\ALL_IN_ONE_RGB_ANOT_COCO\\test\\test_coco.json"
file_train = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT_NEU\Anotations\ALL_IN_ONE_RGB_ANOT_COCO\\train\\train_coco.json"

with open(file_train, "r") as f:
    annot = json.load(f)

new_annot = annot
new_annotations = []
annotations = annot["annotations"]
for anno in annotations:
    segmentation = anno["segmentation"]
    anno["segmentation"] = [segmentation]
    # segm = anno["bbox"]
    new_annotations.append(anno)
new_annot["annotations"] = new_annotations
print()

with open("train_mod_coco.json", "w") as f:
    json.dump(new_annot,f)
