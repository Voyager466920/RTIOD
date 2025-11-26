def train_step(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_samples = 0
    #total_correct = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        features = outputs.unsqueeze(1)

        loss = loss_fn(features, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        #total_correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / total_samples
    #acc = total_correct / total_samples

    return avg_loss



