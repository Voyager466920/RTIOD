import json
import torch
from torch.utils.data import DataLoader

from WorkStation_MoE_3Channel.IRJsonDataset import IRJsonDataset, detection_collate
from WorkStation_MoE_3Channel.MMMMoE.MMMMoE import MMMMoE_Detector

ckpt_path = r"C:\junha\Git\RTIOD\WorkStation_MoE_3Channel\Checkpoints\Resnet34_ColorChannel_model_epoch_11.pt"
json_path = r"C:\junha\Datasets\LTDv2\Valid.json"
image_root = r"C:\junha\Datasets\LTDv2_3Channels\val"
output_pred_path = r"C:\junha\Git\RTIOD\Prediction_Validation\predictions_color_resnet34.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = IRJsonDataset(
    json_path=json_path,
    image_root=image_root,
    require_bbox=False,
)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=detection_collate
)

meta_dim = dataset.meta_dim
model = MMMMoE_Detector(num_classes=5, meta_dim=meta_dim, backbone='resnet34').to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

submission = {}

with torch.no_grad():
    idx = 0
    for images, metas, targets in dataloader:
        images = [img.to(device) for img in images]
        metas = metas.to(device)

        outputs = model(images, metas)

        for i, out in enumerate(outputs):
            image_id = int(targets[i]["image_id"].item())
            uid = str(image_id)

            boxes = out["boxes"].cpu().tolist()
            scores = out["scores"].cpu().tolist()
            labels = out["labels"].cpu().tolist()

            submission[uid] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        idx += len(outputs)

with open(output_pred_path, "w") as f:
    json.dump(submission, f)
