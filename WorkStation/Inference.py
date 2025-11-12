import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
from WorkStation.IRDataset import IRDataset, detection_collate
from Model.AuxDetScratch import AuxDetScratch

weights = r"C:\junha\Git\RTIOD\WorkStation\Checkpoints\model_epoch_001.pt"
csv_path = r"C:\junha\Datasets\LTDv2\metadata_images.csv"
image_root = r"C:\junha\Datasets\LTDv2\mini_Test_frames"
bbox_root = r"C:\junha\Datasets\LTDv2\Train_Labels"
bbox_pattern = "{date}{clip_digits}{frame_digits}.txt"

num_classes = 5
score_thr = 0.01
batch_size = 8
output_dir = r"C:\junha\Git\RTIOD\WorkStation\inference_vis_hasbox"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = IRDataset(csv_path=csv_path, image_root=image_root, bbox_root=bbox_root, bbox_pattern=bbox_pattern)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

model = AuxDetScratch(meta_in_dim=dataset.meta_dim, meta_hidden=64, meta_out_dim=128, num_classes=num_classes).to(device)
sd = torch.load(weights, map_location=device)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
model.eval()

palette = [(255,0,0),(0,255,0),(0,0,255),(255,165,0),(148,0,211),(0,255,255),(255,0,255),(128,128,0),(0,128,128),(128,0,128)]

idx_global = 0
shown = 0
with torch.inference_mode():
    for images, metas, targets in loader:
        images_dev = [im.to(device) for im in images]
        metas_dev = metas.to(device)
        detections, _ = model(images_dev, metas_dev, targets=None)

        for i, det in enumerate(detections):
            boxes = det.get("boxes", torch.zeros((0,4), device=device)).detach().cpu()
            scores = det.get("scores", torch.zeros((0,), device=device)).detach().cpu()
            labels = det.get("labels", torch.zeros((0,), dtype=torch.long, device=device)).detach().cpu()

            keep = (scores >= score_thr).nonzero(as_tuple=False).flatten()
            if keep.numel() == 0:
                idx_global += 1
                continue

            boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
            print(f"[img {idx_global}] kept={len(scores)} max={float(scores.max()):.3f}")

            pil = to_pil_image(images[i].cpu()).convert("RGB")
            W, H = pil.size
            draw = ImageDraw.Draw(pil)

            for j in range(len(scores)):
                x1,y1,x2,y2 = [float(t) for t in boxes[j]]
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                if x2 <= x1 or y2 <= y1: continue
                cidx = max(0, int(labels[j].item()) - 1)
                c = palette[cidx % len(palette)]
                draw.rectangle([x1,y1,x2,y2], outline=c, width=2)
                txt = f"{int(labels[j].item())}:{scores[j]:.2f}"
                x0,y0,x3,y3 = draw.textbbox((0,0), txt)
                tw, th = x3 - x0, y3 - y0
                draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=c)
                draw.text((x1 + 3, y1 - th - 3), txt, fill=(0,0,0))

            out_path = os.path.join(output_dir, f"pred_{idx_global:06d}.png")
            pil.save(out_path)

            # 파이참/윈도우에서 바로 열기
            if os.name == "nt":
                try: os.startfile(out_path)
                except Exception as e: print(f"open failed: {e}", file=sys.stderr)

            shown += 1
            idx_global += 1

print(f"saved & opened {shown} images with boxes to: {output_dir}")
