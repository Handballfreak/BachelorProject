import json

with open("../data/FLIR_ADAS_1_3/train/annotations_allgemein.json", "r") as jso:
    dict = json.loads(jso.read())

cat = dict["categories"]
len_cat = len(cat)
print()