from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from WorkStation_FasterRCNN.Utils import eval_map
from WorkStation_FasterRCNN.IRNoMetaDataset import IRJsonDataset, detection_collate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
from WorkStation_FasterRCNN.Train_Step import train_step


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
    epochs = 40
    batch_size = 32
    initial_lr = 1e-3
    min_lr = 1e-5
    warmup_epochs = 2
    num_classes = 5

    train_losses = []
    mAP50_list = []
    mAP50_95_list = []
    best_epoch = -1
    best_map50 = -1

    train_json = r"C:\junha\Datasets\LTDv2\Train_train.json"
    val_json = r"C:\junha\Datasets\LTDv2\Train_val.json"
    image_root = r"C:\junha\Datasets\LTDv2\frames\frames"

    train_dataset = IRJsonDataset(json_path=train_json, image_root=image_root, require_bbox=True)
    test_dataset = IRJsonDataset(json_path=val_json, image_root=image_root, require_bbox=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=min_lr)

    for epoch in tqdm(range(epochs)):

        train_loss = train_step(train_dataloader, model, optimizer, device)
        metrics_all, _ = eval_map(test_dataloader, model, device, iou_ths=(0.5,), return_per_class=True)

        mAP50 = metrics_all["mAP@0.50"]
        mAP50_95 = metrics_all["mAP@[0.50:0.95]"]

        train_losses.append(train_loss)
        mAP50_list.append(mAP50)
        mAP50_95_list.append(mAP50_95)

        if mAP50 > best_map50:
            best_map50 = mAP50
            best_epoch = epoch

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {train_loss:.4f} | mAP50: {mAP50:.4f} | mAP50-95: {mAP50_95:.4f} | LR: {current_lr:.6f}")

        torch.save(model.state_dict(),fr"C:\junha\Git\RTIOD\WorkStation_MoE\Checkpoints_FasterRCNN\model_epoch_{epoch+1:02d}.pt")

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(mAP50_list, label="mAP@0.50")
    plt.plot(mAP50_95_list, label="mAP@[0.50:0.95]")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.savefig("training_curves.png", dpi=200)
    plt.close()

    print(f"Best Epoch = {best_epoch}")
    print(f"Best mAP50 = {best_map50:.4f}")



if __name__=="__main__":
    main()