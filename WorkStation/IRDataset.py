import os
import csv
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor

def _digits_from_clip(s: str) -> str:
    d = ''.join(ch for ch in s if ch.isdigit())
    return d[-5:] if len(d) >= 5 else d.zfill(5)

def _digits_from_image(s: str) -> str:
    if s.startswith('image_'):
        return s.split('image_')[-1]
    d = ''.join(ch for ch in s if ch.isdigit())
    return d[-4:].zfill(4)

class IRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        bbox_root: Optional[str] = None,
        bbox_pattern: str = "{date}{clip_digits}{frame_digits}.txt",
        date_col: str = "Folder name",
        clip_col: str = "Clip Name",
        frame_col: str = "Image Number",
        force_size: Tuple[int, int] = (288, 384),
    ):
        super().__init__()
        self.image_root = image_root
        self.bbox_root = bbox_root or image_root
        self.bbox_pattern = bbox_pattern
        self.date_col, self.clip_col, self.frame_col = date_col, clip_col, frame_col
        self.force_size = force_size

        with open(csv_path, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        id_cols = {self.date_col, self.clip_col, self.frame_col, "label", "xmin", "ymin", "xmax", "ymax"}
        meta_cols: List[str] = []
        if rows:
            for k in rows[0].keys():
                if k in id_cols:
                    continue
                try:
                    float(rows[0][k]); meta_cols.append(k)
                except: pass
        self.meta_cols = meta_cols

        meta_matrix: List[List[float]] = []
        for r in rows:
            vec = []
            for m in self.meta_cols:
                try: vec.append(float(r[m]))
                except: vec.append(0.0)
            meta_matrix.append(vec)
        if len(meta_matrix) == 0:
            self.meta_min = torch.zeros(0)
            self.meta_max = torch.ones(0)
        else:
            mm = torch.tensor(meta_matrix, dtype=torch.float32)
            self.meta_min = mm.min(dim=0).values
            self.meta_max = mm.max(dim=0).values
        self.eps = 1e-8

        self.samples: List[Dict] = []
        for r in rows:
            date = str(r[self.date_col]).strip()
            clip = str(r[self.clip_col]).strip()
            frame = str(r[self.frame_col]).strip()

            img_path = os.path.join(self.image_root, date, clip, frame + ".jpg")

            raw_meta = []
            for m in self.meta_cols:
                try: raw_meta.append(float(r[m]))
                except: raw_meta.append(0.0)
            if len(raw_meta) > 0:
                v = torch.tensor(raw_meta, dtype=torch.float32)
                den = (self.meta_max - self.meta_min).clamp_min(self.eps)
                v = (v - self.meta_min) / den
            else:
                v = torch.zeros(0, dtype=torch.float32)

            clip_digits = _digits_from_clip(clip)
            frame_digits = _digits_from_image(frame)
            txt_name = self.bbox_pattern.format(date=date, clip=clip, clip_digits=clip_digits, frame=frame, frame_digits=frame_digits)
            bbox_path = os.path.join(self.bbox_root, txt_name)

            self.samples.append({"img_path": img_path, "bbox_path": bbox_path, "meta": v})

        self.meta_dim = len(self.meta_cols)
        #print(f"[IRDataset] meta_cols={len(self.meta_cols)} {self.meta_cols}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str):
        img = Image.open(path).convert('L')
        if self.force_size is not None:
            H, W = self.force_size
            if img.size != (W, H):
                img = img.resize((W, H))
        return to_tensor(img)

    @staticmethod
    def _parse_bbox_txt(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if not os.path.exists(path):
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
        boxes, labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5: continue
                try:
                    c = int(p[0]); x = float(p[1]); y = float(p[2]); w = float(p[3]); h = float(p[4])
                    boxes.append([x, y, x+w, y+h]); labels.append(c+1)
                except: continue
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = self._load_image(s["img_path"])
        boxes, labels = self._parse_bbox_txt(s["bbox_path"])
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, s["meta"], target

def detection_collate(batch):
    images, metas, targets = zip(*batch)
    images = list(images)
    metas = torch.stack(metas, dim=0) if metas and metas[0].numel() > 0 else torch.zeros((len(images), 0), dtype=torch.float32)
    targets = list(targets)
    return images, metas, targets
