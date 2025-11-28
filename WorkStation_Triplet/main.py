import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from SupContrast import SupConLoss
from Model import SupConResNet50
from WorkStation_SupCon.Batch_Sampler import BalancedBatchSampler
from WorkStation_SupCon.Test_Step import test_step
from WorkStation_SupCon.Train_Step import train_step


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 1e-5
    epochs = 10
    batch_size = 64
    num_classes = 4

    base_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform = TwoCropTransform(base_transform)

    train_dataset_path = r"C:\junha\Datasets\LTDv2_patches_train"
    test_dataset_path = r"C:\junha\Datasets\LTDv2_patches_val"

    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=transform)

    train_sampler = BalancedBatchSampler(train_dataset.targets, batch_size, num_classes)

    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model = SupConResNet50(feat_dim=128, pretrained=True).to(device)
    loss_fn = SupConLoss(temperature=0.07).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
        test_loss = test_step(model, test_dataloader, loss_fn, device)

        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        torch.save(
            model.backbone.state_dict(),
            fr"C:\junha\Git\RTIOD\WorkStation_Triplet\Checkpoints\View2\resnet50_supcon_backbone_epoch{epoch + 1}.pth"
        )

    print("\nTraining complete. Final checkpoint saved.")


if __name__ == "__main__":
    main()
