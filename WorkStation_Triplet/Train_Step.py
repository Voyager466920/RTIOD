import torch

def train_step(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for images, labels in dataloader:
        img1, img2 = images
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)

        z1 = model(img1)
        z2 = model(img2)
        features = torch.stack([z1, z2], dim=1)

        loss = loss_fn(features, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = img1.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    avg_loss = total_loss / total_samples
    return avg_loss
