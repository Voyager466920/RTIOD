# Test_Step.py

import torch

@torch.no_grad()
def test_step(dataloader, model, device):
    model.eval()

    total_detected = 0
    total_images = 0

    for images, metas, targets in dataloader:
        images = [img.to(device) for img in images]
        metas = metas.to(device)

        detections = model(images, metas)  # eval → detection list

        for det in detections:
            total_detected += len(det["boxes"])
        total_images += len(images)

    avg_det = total_detected / max(1, total_images)
    return {"avg_detections": avg_det}
