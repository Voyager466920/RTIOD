import os
from WorkStation_MoE.IRJsonDataset import IRJsonDataset

ds = IRJsonDataset(
    json_path=r"C:\junha\Datasets\LTDv2\Train.json",
    image_root=r"C:\junha\Datasets\LTDv2\frames\frames",
    require_bbox=True
)

print("meta_cols:", ds.meta_cols)
print("meta_dim:", ds.meta_dim)
print("num_samples:", len(ds))

missing_images = 0
for s in ds.samples:
    if not os.path.exists(s["img_path"]):
        missing_images += 1

print("missing_images:", missing_images)

img, meta, target = ds[0]
print("img_shape:", img.shape)
print("meta:", meta)
print("boxes:", target["boxes"])
print("labels:", target["labels"])
print("img_path:", ds.samples[0]["img_path"])
print("image_id:", ds.samples[0]["image_id"])
