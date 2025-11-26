import os
import json
from datetime import datetime
from PIL import Image

json_path = r"C:\junha\Datasets\LTDv2\Train_val.json"
dataset_root = r"C:\junha\Datasets\LTDv2\frames"
out_root = r"C:\junha\Datasets\LTDv2_patches_val"

os.makedirs(out_root, exist_ok=True)

max_person_per_bin = 800

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annos = data["annotations"]
categories = data["categories"]

image_dict = {img["id"]: img for img in images}
catid_to_name = {c["id"]: c["name"] for c in categories}

target_names = {"person", "bicycle", "motorcycle", "vehicle"}

person_bin_counter = {}

def sun_bin(s):
    if s <= 0:
        return 0
    if s <= 200:
        return 1
    if s <= 600:
        return 2
    return 3

def temp_bin(t):
    if t < 0:
        return 0
    if t < 10:
        return 1
    if t < 20:
        return 2
    return 3

def hour_bin(h):
    if h < 6:
        return 0
    if h < 12:
        return 1
    if h < 18:
        return 2
    return 3

def month_bin(m):
    if m in (12, 1, 2):
        return 0
    if m in (3, 4, 5):
        return 1
    if m in (6, 7, 8):
        return 2
    return 3

total_ann = len(annos)
saved = 0
saved_person = 0
saved_other = 0

print(f"total annotations: {total_ann}")
print(f"example file_name: {images[0]['file_name']}")
print(f"example img_path: {os.path.join(dataset_root, images[0]['file_name'])}")

for anno in annos:
    img_id = anno["image_id"]
    if img_id not in image_dict:
        continue

    cat_id = int(anno["category_id"])
    if cat_id not in catid_to_name:
        continue

    raw_name = catid_to_name[cat_id]
    cls_name = raw_name.lower()
    if cls_name not in target_names:
        continue

    img_info = image_dict[img_id]
    file_name = img_info["file_name"]
    img_path = os.path.join(dataset_root, file_name)

    if not os.path.exists(img_path):
        continue

    meta = img_info["meta"]
    temp = float(meta["Temperature"])
    hum = float(meta["Humidity"])
    sun = float(meta["Sun Radiation Intensity"])

    try:
        dt = datetime.fromisoformat(img_info["date_captured"])
    except ValueError:
        try:
            dt = datetime.strptime(img_info["date_captured"], "%Y-%m-%dT%H:%M:%S")
        except:
            continue

    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M%S")
    hour = dt.hour
    month = dt.month

    sun_code = int(round(sun))
    temp_code = int(round((temp + 30.0) * 10.0))
    hum_code = int(round(hum))
    hour_code = f"{hour:02d}"
    month_code = f"{month:02d}"

    if cls_name == "person":
        sb = sun_bin(sun)
        tb = temp_bin(temp)
        hb = hour_bin(hour)
        mb = month_bin(month)
        key = (sb, tb, hb, mb)
        c = person_bin_counter.get(key, 0)
        if c >= max_person_per_bin:
            continue
        person_bin_counter[key] = c + 1

    x, y, w, h = anno["bbox"]
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(x + w)
    y2 = int(y + h)

    try:
        img = Image.open(img_path).convert("L")
    except:
        continue

    W, H = img.size
    x1 = max(0, min(x1, W - 1))
    x2 = max(1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(1, min(y2, H))
    if x2 <= x1 or y2 <= y1:
        continue

    crop = img.crop((x1, y1, x2, y2))

    cls_dir = os.path.join(out_root, cls_name)
    os.makedirs(cls_dir, exist_ok=True)

    ann_id = int(anno["id"])

    patch_name = (
        f"{cls_name}_{date_str}_{time_str}_"
        f"s{sun_code}_t{temp_code}_u{hum_code}_"
        f"h{hour_code}_m{month_code}_"
        f"img{img_id}_ann{ann_id}.jpg"
    )

    patch_path = os.path.join(cls_dir, patch_name)

    try:
        crop.save(patch_path)
    except:
        continue

    saved += 1
    if cls_name == "person":
        saved_person += 1
    else:
        saved_other += 1

    if saved % 1000 == 0:
        print(f"[{saved}] saved, person={saved_person}, other={saved_other}")

print("done.")
print(f"saved total : {saved}")
print(f"saved person: {saved_person}")
print(f"saved other : {saved_other}")
print(f"unique person bins: {len(person_bin_counter)}")
