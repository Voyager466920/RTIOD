import torch

def test_step(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    total_correct = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            total_correct += (labels == outputs.argmax(1)).sum().item()

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc