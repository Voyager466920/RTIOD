import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor


class IRJsonDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: Optional[str] = None,
        force_size: Tuple[int, int] = (288, 384),
        only_existing: bool = True,
        require_bbox: bool = True,
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.force_size = force_size
        self.only_existing = only_existing
        self.require_bbox = require_bbox

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

        samples = []
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

            samples.append(
                {
                    "image_id": img_id,
                    "img_path": img_path,
                }
            )

        self.boxes_by_image = boxes_by_image
        self.samples = samples
        self.meta_cols = []
        self.meta_dim = 0

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
        image = self._load_image(s["img_path"])
        image_id = s["image_id"]
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
        return image, target


def detection_collate(batch):
    images, targets = zip(*batch)
    images = list(images)
    targets = list(targets)
    return images, targets
