import torch

@torch.no_grad()
def test_step(dataloader, model, device):
    prev_mode = model.training
    model.train()

    total_loss = 0.0
    total_batches = 0

    for images, metas, targets in dataloader:
        images = [img.to(device) for img in images]
        metas = metas.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, metas, targets)
        loss = sum(v for v in losses.values())

        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_batches += 1

    if not prev_mode:
        model.eval()

    return total_loss / max(1, total_batches)