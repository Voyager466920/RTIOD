import os
import torch
from PIL import Image, ImageDraw

from WorkStation_MoE.IRJsonDataset import IRJsonDataset
from WorkStation_MoE.MMMMoE.MMMMoE import MMMMoE_Detector


def run_inference_and_visualize(
    ckpt_path,
    json_path,
    image_root,
    save_dir,
    idx=0,
    score_thresh=0.5,
):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = IRJsonDataset(
        json_path=json_path,
        image_root=image_root,
        require_bbox=False,
    )

    meta_dim = dataset.meta_dim
    model = MMMMoE_Detector(num_classes=5, meta_dim=meta_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    img_tensor, meta, _ = dataset[idx]
    img_path = dataset.samples[idx]["img_path"]

    images = [img_tensor.to(device)]
    metas = meta.unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model(images, metas)

    det = detections[0]
    boxes = det["boxes"].detach().cpu()
    scores = det["scores"].detach().cpu()
    labels = det["labels"].detach().cpu()

    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if img_tensor.shape[-2:] != (h, w):
        img = img.resize((img_tensor.shape[-1], img_tensor.shape[-2]))
        w, h = img.size

    draw = ImageDraw.Draw(img)
    thickness = 2

    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=thickness)

    base_name = os.path.basename(img_path)
    out_path = os.path.join(save_dir, f"vis_{idx:05d}_{base_name}")
    img.save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    ckpt_path = r"C:\junha\Git\RTIOD\WorkStation_MoE\Checkpoints\model_epoch_15.pt"
    json_path = r"C:\junha\Datasets\LTDv2\Valid.json"
    image_root = r"C:\junha\Datasets\LTDv2\frames\frames"
    save_dir = r"C:\junha\Git\RTIOD\WorkStation_MoE\Inference_Result"

    run_inference_and_visualize(
        ckpt_path=ckpt_path,
        json_path=json_path,
        image_root=image_root,
        save_dir=save_dir,
        idx=0,
        score_thresh=0.5,
    )
