import torch

@torch.inference_mode()
def test_step(dataloader, model, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for images, metas, targets in dataloader:
        images = [img.to(device) for img in images]
        metas = metas.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        _, losses = model(images, metas, targets)
        loss = sum(losses.values())

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)
