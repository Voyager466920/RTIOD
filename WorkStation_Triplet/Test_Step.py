import torch

def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    #total_correct =0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            features = outputs.unsqueeze(1)

            loss = loss_fn(features, labels)

            total_loss += loss.item() * images.size(0)
            #total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    #acc = total_correct / total_samples

    return avg_loss
