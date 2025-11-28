import os
import json
from datetime import datetime
import math
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import numpy as np

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN


JSON_PATH = r"C:\junha\Datasets\LTDv2\Train_train.json"
IMAGE_ROOT = r"C:\junha\Datasets\LTDv2\frames"
CHECKPOINT_DIR = r"C:\junha\Git\RTIOD\VibeCoding_Checkpoint"

BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 4
NUM_CLASSES = 5  # 배경 포함, 객체 클래스 4개면 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METADATA_KEYS = [
    "Temperature",
    "Humidity",
    "Precipitation latest 10 min",
    "Dew Point",
    "Wind Direction",
    "Wind Speed",
    "Sun Radiation Intensity",
    "Min of sunshine latest 10 min",
]


def build_time_embedding(dt):
    hour = dt.hour + dt.minute / 60.0
    month = dt.month
    doy = dt.timetuple().tm_yday

    hour_sin = math.sin(2 * math.pi * hour / 24.0)
    hour_cos = math.cos(2 * math.pi * hour / 24.0)
    month_sin = math.sin(2 * math.pi * (month - 1) / 12.0)
    month_cos = math.cos(2 * math.pi * (month - 1) / 12.0)
    doy_sin = math.sin(2 * math.pi * (doy - 1) / 366.0)
    doy_cos = math.cos(2 * math.pi * (doy - 1) / 366.0)

    return np.array(
        [hour_sin, hour_cos, month_sin, month_cos, doy_sin, doy_cos],
        dtype=np.float32,
    )


class LTDMetaDataset(Dataset):
    def __init__(self, json_path, image_root):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.image_root = image_root
        self.images = data["images"]
        self.annotations = data["annotations"]

        self.id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.id_to_anns:
                self.id_to_anns[img_id] = []
            self.id_to_anns[img_id].append(ann)

        self.meta_mean, self.meta_std = self.compute_meta_stats()

    def compute_meta_stats(self):
        metas = []
        for img in self.images:
            meta = img.get("meta", {})
            vals = [float(meta[k]) for k in METADATA_KEYS]
            metas.append(vals)
        metas = np.array(metas, dtype=np.float32)
        mean = metas.mean(axis=0)
        std = metas.std(axis=0)
        std[std == 0] = 1.0
        return mean, std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.image_root, file_name)

        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        img_id = img_info["id"]
        anns = self.id_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)

        meta_raw = img_info.get("meta", {})
        meta_vals = [float(meta_raw[k]) for k in METADATA_KEYS]
        meta_vals = np.array(meta_vals, dtype=np.float32)
        meta_norm = (meta_vals - self.meta_mean) / self.meta_std

        dt = datetime.fromisoformat(img_info["date_captured"])
        time_embed = build_time_embedding(dt)

        meta_full = np.concatenate([meta_norm, time_embed], axis=0)
        meta_tensor = torch.tensor(meta_full, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
            "meta": meta_tensor,
        }

        return image, target


def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets


class MetaFiLMFasterRCNN(GeneralizedRCNN):
    def __init__(self, num_classes, meta_dim, feat_channels=256):
        base = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        backbone = base.backbone
        rpn = base.rpn
        roi_heads = base.roi_heads
        transform = base.transform
        super().__init__(backbone, rpn, roi_heads, transform)

        self.meta_dim = meta_dim
        self.feat_channels = feat_channels
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * feat_channels),
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("targets must be provided in training mode")

        original_images = images
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append(val)

        images, targets = self.transform(images, targets)

        if targets is not None:
            meta_list = [t["meta"] for t in targets]
            meta_batch = torch.stack(meta_list, dim=0).to(images.tensors.device)
        else:
            meta_batch = None

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {0: features}

        if meta_batch is not None:
            film_params = self.meta_encoder(meta_batch)
            gamma = film_params[:, : self.feat_channels]
            beta = film_params[:, self.feat_channels :]
            for k in features.keys():
                x = features[k]
                b, c, h, w = x.shape
                if c == self.feat_channels:
                    g = gamma.view(b, c, 1, 1)
                    b_ = beta.view(b, c, 1, 1)
                    x = g * x + b_
                    features[k] = x

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections


def train():
    dataset = LTDMetaDataset(JSON_PATH, IMAGE_ROOT)
    meta_dim = len(METADATA_KEYS) + 6

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = MetaFiLMFasterRCNN(num_classes=NUM_CLASSES, meta_dim=meta_dim)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        epoch_loss = 0.0
        for images, targets in dataloader:
            images = [img.to(DEVICE) for img in images]
            for t in targets:
                t["boxes"] = t["boxes"].to(DEVICE)
                t["labels"] = t["labels"].to(DEVICE)
                t["image_id"] = t["image_id"].to(DEVICE)
                t["area"] = t["area"].to(DEVICE)
                t["iscrowd"] = t["iscrowd"].to(DEVICE)
                t["meta"] = t["meta"].to(DEVICE)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    train()
