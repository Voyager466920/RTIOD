import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.manifold import TSNE
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 4
batch_size = 64

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
])

dataset_path = r"C:\junha\Datasets\LTDv2_patches_val"
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

ckpt_path = r"C:\junha\Git\RTIOD\WorkStation_Triplet\Checkpoints\View1\resnet50_supcon_backbone_epoch5.pth"
state_dict = torch.load(ckpt_path, map_location=device)

resnet = models.resnet50(weights=None)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.load_state_dict(state_dict, strict=False)

backbone = nn.Sequential(
    resnet.conv1,
    resnet.bn1,
    resnet.relu,
    resnet.maxpool,
    resnet.layer1,
    resnet.layer2,
    resnet.layer3,
    resnet.layer4,
).to(device)
backbone.eval()

features_list = []
labels_list = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        feats = backbone(images)
        h = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        features_list.append(h.cpu())
        labels_list.append(labels)

features = torch.cat(features_list, dim=0).numpy()
labels = torch.cat(labels_list, dim=0).numpy()

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, metric="euclidean", random_state=0)
features_2d = tsne.fit_transform(features)

plt.figure(figsize=(8, 8))
for c in range(num_classes):
    idx = labels == c
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], s=5, label=str(c))
plt.legend()
plt.tight_layout()
plt.savefig("tsne_visualization.png", dpi=300)
plt.show()
