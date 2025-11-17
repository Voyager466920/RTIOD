import os
from WorkStation_MoE.IRDataset import IRDataset

csv_path = r"C:\junha\Datasets\LTDv2\metadata_images.csv"
image_root = r"C:\junha\Datasets\LTDv2\frames\frames"
bbox_root = r"C:\junha\Datasets\LTDv2\Train_Labels"

ds = IRDataset(
    csv_path=csv_path,
    image_root=image_root,
    bbox_root=bbox_root,
    require_bbox=True
)

print("meta_cols:", ds.meta_cols)
print("meta_dim:", ds.meta_dim)
print("num_samples:", len(ds))

bad_img = 0
bad_box = 0

for s in ds.samples:
    if not os.path.exists(s["img_path"]):
        bad_img += 1
    if not os.path.exists(s["bbox_path"]):
        bad_box += 1

print("missing_images:", bad_img)
print("missing_boxes:", bad_box)

img, meta, target = ds[0]
print("img_shape:", img.shape)
print("meta:", meta)
print("boxes:", target["boxes"])
print("labels:", target["labels"])
print("img_path:", ds.samples[0]["img_path"])
print("bbox_path:", ds.samples[0]["bbox_path"])
