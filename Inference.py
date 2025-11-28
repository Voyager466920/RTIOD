import os
import json
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from Vibecoding import (
    LTDMetaDataset,
    MetaFiLMFasterRCNN,
    METADATA_KEYS,
    collate_fn,
    NUM_CLASSES,
    DEVICE,
    IMAGE_ROOT,
)

JSON_PATH = r"C:\junha\Datasets\LTDv2\train_val.json"
CHECKPOINT_PATH = r"C:\junha\Git\RTIOD\WorkStation_Triplet\Checkpoints\epoch_10.pth"
BATCH_SIZE = 4
NUM_WORKERS = 4
OUTPUT_JSON = r"C:\junha\Git\RTIOD\WorkStation_Triplet\inference_train_val.json"


def load_model(weight_path):
    meta_dim = len(METADATA_KEYS) + 6
    model = MetaFiLMFasterRCNN(num_classes=NUM_CLASSES, meta_dim=meta_dim)
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference():
    dataset = LTDMetaDataset(JSON_PATH, IMAGE_ROOT)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    model = load_model(CHECKPOINT_PATH)

    results = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = [img.to(DEVICE) for img in images]
            for t in targets:
                t["boxes"] = t["boxes"].to(DEVICE)
                t["labels"] = t["labels"].to(DEVICE)
                t["image_id"] = t["image_id"].to(DEVICE)
                t["area"] = t["area"].to(DEVICE)
                t["iscrowd"] = t["iscrowd"].to(DEVICE)
                t["meta"] = t["meta"].to(DEVICE)

            outputs = model(images, targets)

            for out, tgt in zip(outputs, targets):
                img_id = tgt["image_id"].item()
                boxes = out["boxes"].detach().cpu().tolist()
                labels = out["labels"].detach().cpu().tolist()
                scores = out["scores"].detach().cpu().tolist()

                results.append(
                    {
                        "image_id": img_id,
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores,
                    }
                )

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    run_inference()
