import json

file = ""

with open(file, "r") as f:
    annot = json.load(f)

new_annot = annot
new_annotations = []
annotations = annot["annotations"]
for anno in annotations:
    segmentation = anno["segmentation"]
    anno["segmentation"] = [segmentation]
    new_annotations.append(anno)
new_annot["annotations"] = new_annotations

path = file.replace("coco.json", "mod_coco.json")
with open(path, "w") as f:
    json.dump(new_annot,f)
