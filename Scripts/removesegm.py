import json

file_test = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT\\test\\test_coco.json"
file_train = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT\\train\\train_coco.json"

with open(file_test, "r") as f:
    annot = json.load(f)

new_annot = annot
new_annotations = []
annotations = annot["annotations"]
for anno in annotations:
    anno["segmentation"] = []
    # segm = anno["bbox"]
    new_annotations.append(anno)
new_annot["annotations"] = new_annotations
print()

with open("test_mod_coco.json", "w") as f:
    json.dump(new_annot,f)
