import json, os
from collections import defaultdict
from pathlib import Path

json_path = r"C:\junha\Datasets\LTDv2\Train.json"
out_dir = r"C:\junha\Datasets\LTDv2\Train_Labels"
Path(out_dir).mkdir(parents=True, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

anns = data["annotations"] if "annotations" in data else data

by_image = defaultdict(list)
for a in anns:
    if "bbox" not in a:
        continue
    img_id = a["image_id"]
    x, y, w, h = a["bbox"]
    cid = a["category_id"]
    by_image[img_id].append((cid, x, y, w, h))

for img_id, boxes in by_image.items():
    txt_path = os.path.join(out_dir, f"{img_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for cid, x, y, w, h in boxes:
            f.write(f"{cid} {x} {y} {w} {h}\n")

print(f"{len(by_image)} image label files saved in '{out_dir}'")