import json

file_test = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT\\test\\test_coco.json"
file_train = "E:\BA_BP_Datem\Bachelor Project\BachelorProject\data\ALL_IN_ONE_RGB_IMG_ANOT\\train\\train_coco.json"

print("Test")

with open(file_train, "r") as f:
    annot = json.load(f)

annotations = annot["annotations"]
for anno in annotations:
    segm = anno["segmentation"]
    # segm = anno["bbox"]
    if type(segm) is not list or len(segm) == 0:
        id = anno["id"]
        print(id, type(segm))
print()