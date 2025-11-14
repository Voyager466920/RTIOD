import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from WorkStation_AuxDet.IRDataset import IRDataset, detection_collate
from Model.AuxDetScratch import AuxDetScratch

weights = r"C:\junha\Git\RTIOD\WorkStation\Checkpoints\model_epoch_001.pt"
csv_path = r"C:\junha\Datasets\LTDv2\metadata_images.csv"
image_root = r"C:\junha\Datasets\LTDv2\mini_Test_frames"
bbox_root = r"C:\junha\Datasets\LTDv2\Train_Labels"
bbox_pattern = "{date}{clip_digits}{frame_digits}.txt"

num_classes = 5
score_thr = 0.0
batch_size = 8
max_images = 800
stride = 3
output_dir = r"/WorkStation_AuxDet\inference_vis_hasbox"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = IRDataset(csv_path=csv_path, image_root=image_root, bbox_root=bbox_root, bbox_pattern=bbox_pattern)
indices = list(range(0, len(dataset), stride))[:max_images]
subset = torch.utils.data.Subset(dataset, indices)
loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

model = AuxDetScratch(meta_in_dim=dataset.meta_dim, meta_hidden=64, meta_out_dim=128, num_classes=num_classes).to(device)
sd = torch.load(weights, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

rpn = model.rpn.rpn
if hasattr(rpn, "_pre_nms_top_n") and hasattr(rpn, "_post_nms_top_n"):
    rpn._pre_nms_top_n["testing"] = 300
    rpn._post_nms_top_n["testing"] = 100
elif isinstance(getattr(rpn, "pre_nms_top_n", None), dict):
    rpn.pre_nms_top_n["testing"] = 300
    rpn.post_nms_top_n["testing"] = 100

color_box = (255, 0, 0)
line_width = 8

idx_global = 0
saved = 0

with torch.inference_mode():
    for images, metas, targets in loader:
        images_dev = [im.to(device) for im in images]
        metas_dev = metas.to(device)
        detections, _ = model(images_dev, metas_dev, targets=None)

        for i, det in enumerate(detections):
            boxes = det.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu()
            scores = det.get("scores", torch.zeros((0,), device=device)).detach().cpu()
            labels = det.get("labels", torch.zeros((0,), dtype=torch.long, device=device)).detach().cpu()

            keep = (scores >= score_thr).nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                idx_global += 1
                continue

            boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]

            pil = to_pil_image(images[i].cpu()).convert("RGB")
            W, H = pil.size
            pil = pil.convert("RGBA")
            draw = ImageDraw.Draw(pil, "RGBA")

            boxes_draw = boxes.clone()
            if boxes_draw.numel() > 0 and float(boxes_draw.max()) <= 2.5:
                boxes_draw[:, [0, 2]] *= W
                boxes_draw[:, [1, 3]] *= H

            def clamp_box(x1, y1, x2, y2):
                x1 = max(0, min(float(x1), W - 1))
                x2 = max(0, min(float(x2), W - 1))
                y1 = max(0, min(float(y1), H - 1))
                y2 = max(0, min(float(y2), H - 1))
                return x1, y1, x2, y2

            drawn = 0
            for j in range(len(scores)):
                x1, y1, x2, y2 = boxes_draw[j]
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2)
                if x2 <= x1 or y2 <= y1:
                    cx, cy, w, h = boxes_draw[j]
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                draw.rectangle([x1, y1, x2, y2], outline=(color_box[0], color_box[1], color_box[2], 255), width=line_width)

                txt = f"{int(labels[j].item())}:{scores[j]:.2f}"
                x0, y0, x3, y3 = draw.textbbox((0, 0), txt)
                tw, th = x3 - x0, y3 - y0
                draw.rectangle([x1, y1 - th - 8, x1 + tw + 10, y1], fill=(color_box[0], color_box[1], color_box[2], 220))
                draw.text((x1 + 5, y1 - th - 6), txt, fill=(0, 0, 0, 255))
                drawn += 1

            pil = pil.convert("RGB")
            out_path = os.path.join(output_dir, f"pred_{indices[idx_global]:06d}.png")
            pil.save(out_path)
            print(f"[img {indices[idx_global]}] kept={len(scores)} drawn={drawn} max={float(scores.max()):.3f}")
            saved += 1
            idx_global += 1

print(f"✅ saved {saved} images with bright red outline boxes to: {output_dir}")
