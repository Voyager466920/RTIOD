import torch
from torch.utils.data import DataLoader

from WorkStation_MoE.IRJsonDataset import IRJsonDataset, detection_collate
from WorkStation_MoE.MMMMoE.Original_MMMMoE.MMMMoE import MMMMoE_Detector
from WorkStation_MoE.Utils import eval_map


def validate_model(
        ckpt_path,
        json_path,
        image_root,
        num_classes=5,
        batch_size=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = IRJsonDataset(
        json_path=json_path,
        image_root=image_root,
        require_bbox=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate
    )

    meta_dim = dataset.meta_dim
    model = MMMMoE_Detector(num_classes=num_classes, meta_dim=meta_dim).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"Dataset size: {len(dataset)}")
    print(f"Meta dim: {meta_dim}")
    print(f"Num classes: {num_classes}")
    print("-" * 50)

    metrics_all, per_class = eval_map(
        dataloader,
        model,
        device,
        iou_ths=(0.5,),
        return_per_class=True
    )

    print("Overall Metrics:")
    for key, val in metrics_all.items():
        print(f"  {key}: {val:.4f}")

    print("\nPer-Class Metrics:")
    for cls_id, metrics in per_class.items():
        print(f"  Class {cls_id}:")
        for key, val in metrics.items():
            print(f"    {key}: {val:.4f}")

    return metrics_all, per_class


if __name__ == "__main__":

    ckpt_path = r"C:\junha\Git\RTIOD\WorkStation_MoE\Checkpoints_Workstation\Month_Hour_model_epoch_04.pt"
    json_path = r"C:\junha\Datasets\LTDv2\Valid.json"
    image_root = r"C:\junha\Datasets\LTDv2\frames\frames"

    validate_model(
        ckpt_path=ckpt_path,
        json_path=json_path,
        image_root=image_root,
        num_classes=5,
        batch_size=32,
    )