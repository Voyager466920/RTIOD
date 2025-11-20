import torch
from tqdm.auto import tqdm

def train_step(dataloader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for i, (images, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        losses = model(images, targets)

        if i == 0:
            print("losses dict:")
            for k, v in losses.items():
                print(k, v.detach().cpu().item())

        loss = sum(losses.values())

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)
