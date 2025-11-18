from tqdm.auto import tqdm

import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from WorkStation_MoE.Utils import eval_map
from WorkStation_MoE.IRJsonDataset import IRJsonDataset, detection_collate
from WorkStation_MoE.MMMMoE.MMMMoE import MMMMoE_Detector
from WorkStation_MoE.Test_Step import test_step
from WorkStation_MoE.Train_Step import train_step


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            return [base_lr for base_lr in self.base_lrs]
        warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 100
    batch_size = 64
    initial_lr = 1e-3
    min_lr = 1e-5
    warmup_epochs = 2
    num_classes = 5

    train_json = r"C:\junha\Datasets\LTDv2\mini_train.json"
    test_json = r"C:\junha\Datasets\LTDv2\mini_test.json"
    image_root = r"C:\junha\Datasets\LTDv2\frames\frames"

    train_dataset = IRJsonDataset(json_path=train_json, image_root=image_root, require_bbox=True)
    test_dataset = IRJsonDataset(json_path=test_json, image_root=image_root, require_bbox=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

    meta_dim = train_dataset.meta_dim
    model = MMMMoE_Detector(num_classes=num_classes, meta_dim=meta_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=min_lr)

    for epoch in tqdm(range(epochs)):
        moe = model.backbone.moeblock
        moe.usage_soft.zero_()
        moe.usage_hard.zero_()
        moe.num_batches.zero_()

        train_loss = train_step(train_dataloader, model, optimizer, device)

        metrics_all, per_class = eval_map(test_dataloader,model,device,iou_ths=(0.5,), return_per_class=True)
        mAP50 = metrics_all["mAP@0.50"]
        mAP50_95 = metrics_all["mAP@[0.50:0.95]"]

        total_detected = test_step(test_dataloader, model, device)["avg_detections"]

        print(
            f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"train_loss: {train_loss:.4f} | "
            f"Test(avg_det): {total_detected:.2f} | "
            f"mAP50: {mAP50:.4f} | mAP50:95: {mAP50_95:.4f}"
        )

        print("per-class mAP50:")
        for cid, v in per_class.items():
            print(f"  cls {cid}: {v['mAP@0.50']:.4f}")

        print("per-class mAP50:95:")
        for cid, v in per_class.items():
            print(f"  cls {cid}: {v['mAP@[0.50:0.95]']:.4f}")

        soft = moe.usage_soft
        hard = moe.usage_hard
        soft_freq = (soft / soft.sum().clamp_min(1e-8)).detach().cpu().tolist()
        hard_freq = (hard / hard.sum().clamp_min(1e-8)).detach().cpu().tolist()

        print("expert soft freq:", soft_freq)
        print("expert hard freq:", hard_freq)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        torch.save(model.state_dict(), r"C:/junha/Git/RTIOD/WorkStation_MoE/Checkpoints/model_epoch_{epoch + 1:02d}.pt")
        print(f"saved: model_epoch_{epoch + 1:03d}.pt")


if __name__=="__main__":
    main()