from tqdm.auto import tqdm

import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from WorkStation_AuxDet.IRDataset import detection_collate
from WorkStation_AuxDet.Utils import eval_map
from WorkStation_MoE.IRJsonDataset import IRJsonDataset
from WorkStation_MoE.M4E.M4E import MMMMoE_Detector
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

    print("train_len:", len(train_dataset), "test_len:", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

    meta_dim = train_dataset.meta_dim
    model = MMMMoE_Detector(num_classes=num_classes, meta_dim=meta_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=min_lr)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(train_dataloader, model, optimizer, device)
        test_info = test_step(test_dataloader, model, device)
        metrics = eval_map(test_dataloader, model, device, iou_ths=(0.5,))

        print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"train_loss: {train_loss:.4f} | "
              f"Test(avg_det): {test_info['avg_detections']:.2f} | "
              f"mAP50: {metrics['mAP@0.50']:.4f}")

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        torch.save(model.state_dict(), f"C:\\junha\\Git\\RTIOD\\WorkStation_MoE\\Checkpoints\\model_epoch_{epoch + 1:02d}.pt")
        print(f"saved: model_epoch_{epoch + 1:03d}.pt")


if __name__ == "__main__":
    main()
