import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor

import cv2
import numpy as np
from ultralytics.data.augment import (
    Compose,
    RandomHSV,
    RandomFlip,
    RandomPerspective,
    LetterBox,
)
from ultralytics.utils.instance import Instances
from datetime import datetime


class IRJsonDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: Optional[str] = None,
        force_size: Tuple[int, int] = (288, 384),
        only_existing: bool = True,
        require_bbox: bool = True,
        augment: bool = True,
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.force_size = force_size
        self.only_existing = only_existing
        self.require_bbox = require_bbox
        self.augment = augment

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "images" in data and "annotations" in data:
            images = data["images"]
            anns = data["annotations"]
        else:
            images = data.get("images", [])
            anns = data.get("annotations", data.get("anns", []))

        boxes_by_image: Dict[int, List[Dict]] = defaultdict(list)
        for a in anns:
            if "bbox" not in a or "image_id" not in a:
                continue
            boxes_by_image[a["image_id"]].append(a)

        self.boxes_by_image = boxes_by_image

        new_shape = (self.force_size[0], self.force_size[1])
        letter_box = LetterBox(
            new_shape=new_shape,
            auto=False,
            scaleup=True,
            center=True,
            stride=32,
        )
        if self.augment:
            self.transforms = Compose(
                [
                    RandomPerspective(
                        degrees=0.0,
                        translate=0.1,
                        scale=0.0,
                        shear=0.0,
                        perspective=0.0,
                        border=(0, 0),
                        pre_transform=None,
                    ),
                    RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4),
                    RandomFlip(p=0.5, direction="horizontal"),
                    letter_box,
                ]
            )
        else:
            self.transforms = Compose([letter_box])

        samples = []
        meta_cols_set = None

        for img in images:
            img_id = img["id"]
            file_name = img["file_name"]

            if self.image_root is None:
                if os.path.isabs(file_name):
                    img_path = file_name
                else:
                    img_path = os.path.normpath(file_name)
            else:
                if file_name.startswith("frames/") or file_name.startswith("frames\\"):
                    rel = file_name.split("frames/", 1)[-1].split("frames\\", 1)[-1]
                    img_path = os.path.join(self.image_root, rel)
                else:
                    img_path = os.path.join(self.image_root, file_name)

            if self.only_existing and not os.path.exists(img_path):
                continue

            if self.require_bbox and img_id not in boxes_by_image:
                continue

            meta_dict = img.get("meta", {})
            date_str = img.get("date_captured")
            if isinstance(date_str, str):
                dt = None
                try:
                    dt = datetime.fromisoformat(date_str)
                except ValueError:
                    try:
                        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        dt = None
                if dt is not None:
                    meta_dict["Month"] = dt.month
                    meta_dict["Hour"] = dt.hour + dt.minute / 60.0

            if meta_cols_set is None:
                meta_cols = []
                for k, v in meta_dict.items():
                    if isinstance(v, (int, float)):
                        meta_cols.append(k)
                meta_cols_set = meta_cols
            samples.append(
                {
                    "image_id": img_id,
                    "img_path": img_path,
                    "meta_raw": meta_dict,
                }
            )

        if meta_cols_set is None:
            self.meta_cols = []
            self.meta_dim = 0
            self.samples = []
            self.boxes_by_image = {}
            return

        self.meta_cols = list(meta_cols_set)
        self.meta_dim = len(self.meta_cols)

        mins = {k: float("inf") for k in self.meta_cols}
        maxs = {k: float("-inf") for k in self.meta_cols}

        for s in samples:
            m = s["meta_raw"]
            for k in self.meta_cols:
                v = m.get(k, 0.0)
                if not isinstance(v, (int, float)):
                    v = 0.0
                v = float(v)
                if v < mins[k]:
                    mins[k] = v
                if v > maxs[k]:
                    maxs[k] = v

        for k in self.meta_cols:
            if mins[k] == float("inf"):
                mins[k] = 0.0
            if maxs[k] == float("-inf"):
                maxs[k] = 0.0

        for s in samples:
            m = s["meta_raw"]
            vals = []
            for k in self.meta_cols:
                v = m.get(k, 0.0)
                if not isinstance(v, (int, float)):
                    v = 0.0
                v = float(v)
                lo = mins[k]
                hi = maxs[k]
                if hi > lo:
                    v = (v - lo) / (hi - lo)
                else:
                    v = 0.0
                vals.append(v)
            s["meta"] = torch.tensor(vals, dtype=torch.float32)

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        if self.force_size is not None:
            h, w = self.force_size
            img = img.resize((w, h))
        return to_tensor(img)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = s["img_path"]
        image_id = s["image_id"]
        meta = s["meta"]
        anns = self.boxes_by_image.get(image_id, [])

        boxes_list = []
        labels_list = []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            boxes_list.append([x1, y1, x2, y2])
            cid = int(a["category_id"])
            labels_list.append(cid)

        if self.augment:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

            if len(boxes_list) == 0:
                bboxes_np = np.zeros((0, 4), dtype=np.float32)
                labels_np = np.zeros((0, 1), dtype=np.float32)
            else:
                bboxes_np = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
                labels_np = np.asarray(labels_list, dtype=np.float32).reshape(-1, 1)

            segments = np.zeros((0, 0, 2), dtype=np.float32)

            labels_dict = {
                "img": img,
                "cls": labels_np,
                "instances": Instances(
                    bboxes=bboxes_np,
                    bbox_format="xyxy",
                    normalized=False,
                    segments=segments,
                ),
                "segments": segments,
            }

            labels_dict = self.transforms(labels_dict)

            aug_img = labels_dict["img"]
            instances = labels_dict.get("instances", None)
            if instances is not None:
                boxes_out = instances.bboxes
            else:
                boxes_out = np.zeros((0, 4), dtype=np.float32)

            labels_out = labels_dict.get("cls", np.zeros((0, 1), dtype=np.float32))

            img_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
            image = torch.from_numpy(img_gray).float().unsqueeze(0) / 255.0

            if len(boxes_out):
                boxes = torch.tensor(boxes_out, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)

            if len(labels_out):
                labels = torch.tensor(labels_out.reshape(-1), dtype=torch.int64)
            else:
                labels = torch.zeros((0,), dtype=torch.int64)

        else:
            image = self._load_image(img_path)
            if len(boxes_list) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes_list, dtype=torch.float32)
                labels = torch.tensor(labels_list, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }
        return image, meta, target


def detection_collate(batch):
    images, metas, targets = zip(*batch)
    images = list(images)
    metas = (
        torch.stack(metas, dim=0)
        if metas and metas[0].numel() > 0
        else torch.zeros((len(images), 0), dtype=torch.float32)
    )
    targets = list(targets)
    return images, metas, targets
