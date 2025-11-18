import json
import random

# 경로 설정
train_json_path = r"C:\junha\Datasets\LTDv2\Train.json"
out_train_json_path = r"C:\junha\Datasets\LTDv2\Train_train.json"
out_val_json_path = r"C:\junha\Datasets\LTDv2\Train_val.json"

split_ratio = 0.8   # 8:2
seed = 42           # seed 고정

# Load JSON
with open(train_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", [])
info = data.get("info", {})
licenses = data.get("licenses", [])
categories = data.get("categories", [])

# Shuffle images with fixed seed
random.seed(seed)
random.shuffle(images)

# Compute split index
num_train = int(len(images) * split_ratio)
train_images = images[:num_train]
val_images = images[num_train:]

# Map image IDs
train_ids = {img["id"] for img in train_images}
val_ids = {img["id"] for img in val_images}

# Filter annotations
train_annotations = [a for a in annotations if a["image_id"] in train_ids]
val_annotations = [a for a in annotations if a["image_id"] in val_ids]

# Create split JSONs
train_json = {
    "info": info,
    "licenses": licenses,
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories,
}

val_json = {
    "info": info,
    "licenses": licenses,
    "images": val_images,
    "annotations": val_annotations,
    "categories": categories,
}

# Save
with open(out_train_json_path, "w", encoding="utf-8") as f:
    json.dump(train_json, f, ensure_ascii=False)

with open(out_val_json_path, "w", encoding="utf-8") as f:
    json.dump(val_json, f, ensure_ascii=False)

print("Train images:", len(train_images), " | Annotations:", len(train_annotations))
print("Val images:", len(val_images), " | Annotations:", len(val_annotations))
