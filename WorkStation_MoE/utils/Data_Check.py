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

print("num_samples:", len(ds))

used = set(os.path.basename(s["bbox_path"]) for s in ds.samples)
print("used_label_files:", len(used))

all_labels = [f for f in os.listdir(bbox_root) if f.lower().endswith(".txt")]
print("all_label_files:", len(all_labels))

unused = set(all_labels) - used
print("unused_label_files:", len(unused))

print("example_used:", list(sorted(used))[:10])
print("example_unused:", list(sorted(unused))[:10])
