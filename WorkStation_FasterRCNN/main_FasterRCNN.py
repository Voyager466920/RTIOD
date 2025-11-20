
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from tqdm import tqdm


class CocoDataset(Dataset):
    def __init__(self, root, ann_path, transforms=None):
        self.root = root
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.transforms = transforms

        self.ann_map = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.ann_map:
                self.ann_map[img_id] = []
            self.ann_map[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_id = info["id"]
        img_path = os.path.join(self.root, info["file_name"])
        img = Image.open(img_path).convert("RGB")

        anns = self.ann_map.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            areas.append(a["area"])
            iscrowd.append(a.get("iscrowd", 0))

        if len(boxes) == 0:
            raise ValueError("skip this image")

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        img_id_tensor = torch.tensor([img_id], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
            "image_id": img_id_tensor
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train(model, loader, device, epochs=10, lr=0.005):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, tgts in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            losses = sum(loss_dict.values())
            opt.zero_grad()
            losses.backward()
            opt.step()
            total_loss += losses.item()
        print(epoch + 1, total_loss)

    return model


def box_iou(box, boxes):
    ixmin = np.maximum(box[0], boxes[:, 0])
    iymin = np.maximum(box[1], boxes[:, 1])
    ixmax = np.minimum(box[2], boxes[:, 2])
    iymax = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inter = iw * ih
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    iou = inter / np.maximum(union, 1e-6)
    return iou


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


@torch.no_grad()
def evaluate_map(model, data_loader, device, num_classes, score_thresh=0.05):
    model.eval()
    model.to(device)

    gt_boxes = {c: {} for c in range(1, num_classes)}
    pred_boxes = {c: [] for c in range(1, num_classes)}
    npos = {c: 0 for c in range(1, num_classes)}

    for imgs, tgts in tqdm(data_loader, desc="Evaluating"):
        imgs = [i.to(device) for i in imgs]
        outputs = model(imgs)
        for out, tgt in zip(outputs, tgts):
            img_id = int(tgt["image_id"].item())
            boxes = tgt["boxes"].cpu().numpy()
            labels = tgt["labels"].cpu().numpy()

            for b, l in zip(boxes, labels):
                l = int(l)
                if l == 0 or l >= num_classes:
                    continue
                if img_id not in gt_boxes[l]:
                    gt_boxes[l][img_id] = []
                gt_boxes[l][img_id].append(b)
                npos[l] += 1

            pb = out["boxes"].cpu().numpy()
            pl = out["labels"].cpu().numpy()
            ps = out["scores"].cpu().numpy()

            for b, l, s in zip(pb, pl, ps):
                l = int(l)
                if l == 0 or l >= num_classes:
                    continue
                if s < score_thresh:
                    continue
                pred_boxes[l].append((img_id, s, b))

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    aps_50 = []
    aps_50_95 = []

    for c in range(1, num_classes):
        if npos[c] == 0:
            continue
        preds = pred_boxes[c]
        if len(preds) == 0:
            continue
        preds = sorted(preds, key=lambda x: -x[1])
        scores = np.array([p[1] for p in preds])
        boxes = np.array([p[2] for p in preds])

        gt_for_class = {}
        for img_id, gts in gt_boxes[c].items():
            gt_for_class[img_id] = {
                "boxes": np.array(gts),
                "matched": np.zeros(len(gts), dtype=bool),
            }

        aps_per_iou = []
        for t in iou_thresholds:
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))

            for i, (img_id, s, b) in enumerate(preds):
                if img_id not in gt_for_class:
                    fp[i] = 1
                    continue
                g = gt_for_class[img_id]
                g_boxes = g["boxes"]
                g_matched = g["matched"]

                if g_boxes.shape[0] == 0:
                    fp[i] = 1
                    continue

                ious = box_iou(b, g_boxes)
                jmax = np.argmax(ious)
                iou_max = ious[jmax]

                if iou_max >= t and not g_matched[jmax]:
                    tp[i] = 1
                    g_matched[jmax] = True
                else:
                    fp[i] = 1

            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)
            rec = tp_cum / float(npos[c])
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
            ap = compute_ap(rec, prec)
            aps_per_iou.append(ap)

        aps_per_iou = np.array(aps_per_iou)
        aps_50.append(aps_per_iou[0])
        aps_50_95.append(aps_per_iou.mean())

    mAP_50 = float(np.mean(aps_50)) if len(aps_50) > 0 else 0.0
    mAP_50_95 = float(np.mean(aps_50_95)) if len(aps_50_95) > 0 else 0.0
    return mAP_50, mAP_50_95


if __name__ == "__main__":
    root = r"C:\junha\Datasets\LTDv2\frames"
    ann_train = "C:/junha/Datasets/LTDv2/train_train.json"
    ann_val = "C:/junha/Datasets/LTDv2/train_val.json"

    dataset_train = CocoDataset(root, ann_train, transforms=F.to_tensor)
    dataset_val = CocoDataset(root, ann_val, transforms=F.to_tensor)

    cats_train = {a["category_id"] for a in dataset_train.annotations}
    cats_val = {a["category_id"] for a in dataset_val.annotations}
    num_classes = 1 + len(cats_train.union(cats_val))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes)
    model = train(model, train_loader, device, epochs=10, lr=0.005)

    mAP_50, mAP_50_95 = evaluate_map(model, val_loader, device, num_classes)
    print("mAP@0.5:", mAP_50)
    print("mAP@0.5:0.95:", mAP_50_95)

    torch.save(model.state_dict(), "fasterrcnn_coco.pth")
