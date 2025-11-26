import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from SupContrast import SupConLoss
from Model import SupConResNet50
from WorkStation_Triplet.Batch_Sampler import BalancedBatchSampler
from WorkStation_Triplet.Test_Step import test_step
from WorkStation_Triplet.Train_Step import train_step


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 1e-4
    epochs = 10
    batch_size = 64
    num_classes = 4

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    train_dataset_path = r"C:\junha\Datasets\LTDv2_patches_train"
    test_dataset_path = r"C:\junha\Datasets\LTDv2_patches_val"

    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)
    train_sampler = BalancedBatchSampler(train_dataset.targets, batch_size, num_classes)
    test_sampler = BalancedBatchSampler(train_dataset.targets, batch_size, num_classes)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, pin_memory=True)

    model = SupConResNet50(feat_dim=128, pretrained=True).to(device)
    loss_fn = SupConLoss(temperature=0.07).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
        test_loss = test_step(model, test_dataloader, loss_fn, device)

        print(f"\nEpoch [{epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        torch.save(model.encoder.state_dict(),f"resnet50_supcon_backbone_epoch{epoch+1}.pth")

    print("\nTraining complete. Final checkpoint saved.")


if __name__ == "__main__":
    main()
