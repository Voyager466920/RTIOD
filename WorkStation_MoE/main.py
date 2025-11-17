from tqdm.auto import tqdm

import torch.cuda
import torch.optim as optim
from torch.utils.data import DataLoader

from WorkStation_AuxDet.IRDataset import detection_collate
from WorkStation_AuxDet.Utils import eval_map
from WorkStation_MoE.IRDataset import IRDataset
from WorkStation_MoE.M4E.M4E import MMMMoE_Detector
from WorkStation_MoE.Test_Step import test_step
from WorkStation_MoE.Train_Step import train_step


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 10
    batch_size = 64
    lr = 1e-4
    num_classes = 5

    csv_path = r"C:\junha\Datasets\LTDv2\metadata_images.csv"
    #image_root = r"C:\junha\Datasets\LTDv2\frames\frames"
    train_image_root = r"C:\junha\Datasets\LTDv2\mini_Train_frames"
    test_image_root = r"C:\junha\Datasets\LTDv2\mini_Test_frames"
    bbox_root = r"C:\junha\Datasets\LTDv2\Train_Labels"
    bbox_pattern = "{date}{clip_digits}{frame_digits}.txt"

    train_dataset = IRDataset(csv_path=csv_path, image_root=train_image_root, bbox_root=bbox_root, bbox_pattern=bbox_pattern)
    test_dataset = IRDataset(csv_path=csv_path, image_root=test_image_root, bbox_root=bbox_root, bbox_pattern=bbox_pattern)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate)
    meta_dim = train_dataset.meta_dim
    model = MMMMoE_Detector(num_classes=num_classes,meta_dim=meta_dim,).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(train_dataloader, model, optimizer, device)
        test_info  = test_step(test_dataloader, model, device)
        metrics = eval_map(test_dataloader, model, device, iou_ths=(0.5,))
        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} |Test(avg_det): {test_info['avg_detections']:.2f} | mAP50: {metrics['mAP@0.50']:.4f}")
        torch.save(model.state_dict(), f"C:\junha\Git\RTIOD\WorkStation_MoE\Checkpoints\model_epoch_{epoch + 1:02d}.pt")
        print(f"saved: model_epoch_{epoch + 1:03d}.pt")

if __name__=="__main__":
    main()