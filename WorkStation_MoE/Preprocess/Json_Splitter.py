import json
import random

train_json_path = r"C:\junha\Datasets\LTDv2\Train.json"
out_train_json_path = r"C:\junha\Datasets\LTDv2\mini_train.json"
out_test_json_path = r"C:\junha\Datasets\LTDv2\mini_test.json"

num_train = 12000
num_test = 1200

with open(train_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", [])
info = data.get("info", {})
licenses = data.get("licenses", [])
categories = data.get("categories", [])

random.seed(42)
random.shuffle(images)

sel_train_images = images[:num_train]
sel_test_images = images[num_train:num_train + num_test]

train_ids = set(img["id"] for img in sel_train_images)
test_ids = set(img["id"] for img in sel_test_images)

train_annotations = [a for a in annotations if a.get("image_id") in train_ids]
test_annotations = [a for a in annotations if a.get("image_id") in test_ids]

mini_train = {
    "info": info,
    "licenses": licenses,
    "images": sel_train_images,
    "annotations": train_annotations,
    "categories": categories,
}

mini_test = {
    "info": info,
    "licenses": licenses,
    "images": sel_test_images,
    "annotations": test_annotations,
    "categories": categories,
}

with open(out_train_json_path, "w", encoding="utf-8") as f:
    json.dump(mini_train, f, ensure_ascii=False)

with open(out_test_json_path, "w", encoding="utf-8") as f:
    json.dump(mini_test, f, ensure_ascii=False)

print("mini_train images:", len(sel_train_images), "anns:", len(train_annotations))
print("mini_test images:", len(sel_test_images), "anns:", len(test_annotations))
