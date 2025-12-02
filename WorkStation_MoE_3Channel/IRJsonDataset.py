import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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
        meta_cols_set = None

        for img in images:
            img_id = img["id"]
            # JSON에 있는 원래 파일 경로 (예: frames/20200828/clip_6_0248/image_0048.jpg)
            original_file_name = img["file_name"]

            # ---------------------------------------------------------
            # [수정된 부분] 실제 디스크의 파일명 형식에 맞게 변환
            # 1. 경로 구분자(/, \)를 언더바(_)로 변경
            # 2. 확장자 .jpg를 .png로 변경 (필요한 경우)
            # ---------------------------------------------------------
            flat_file_name = original_file_name.replace("/", "_").replace("\\", "_")
            flat_file_name = flat_file_name.replace(".jpg", ".png")

            if self.image_root is None:
                # image_root가 없으면 현재 경로 기준 (사용자가 절대경로를 줬으므로 여기론 안 올 것임)
                img_path = flat_file_name
            else:
                # 지정된 root 경로와 플랫한 파일명 결합
                img_path = os.path.join(self.image_root, flat_file_name)

            # 디버깅용: 파일이 없으면 경로가 맞는지 한번 확인해보기 위해 출력 가능
            # if not os.path.exists(img_path):
            #     print(f"File not found: {img_path}")

            if self.only_existing and not os.path.exists(img_path):
                continue

            if self.require_bbox and img_id not in boxes_by_image:
                continue

            # 메타데이터 파싱 (기존 로직 유지)
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
        self.boxes_by_image = boxes_by_image

        # 메타데이터 정규화 (기존 로직 유지)
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
        # 1채널(L)로 로드, 필요시 3채널(RGB)로 변경 가능
        img = Image.open(path).convert("L")
        if self.force_size is not None:
            h, w = self.force_size
            img = img.resize((w, h))
        return to_tensor(img)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = self._load_image(s["img_path"])
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
    metas = torch.stack(metas, dim=0) if metas and metas[0].numel() > 0 else torch.zeros((len(images), 0), dtype=torch.float32)
    targets = list(targets)
    return images, metas, targets
