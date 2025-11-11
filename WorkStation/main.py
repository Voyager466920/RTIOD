from tqdm.auto import tqdm

import torch.cuda
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from Model.AuxDetScratch import AuxDetScratch
from WorkStation.IRDataset import IRDataset
from WorkStation.Test_Step import test_step
from WorkStation.Train_Step import train_step


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 100
    batch_size = 128
    lr = 1e-4
    num_classes = 5

    csv_path = r"C:\junha\Datasets\LTDv2\metadata_images.csv"
    image_root = r"C:\junha\Datasets\LTDv2\frames\frames"
    bbox_root = r"C:\junha\Datasets\LTDv2\Train_Labels"
    bbox_pattern = "{date}{clip_digits}{frame_digits}.txt"

    train_dataset = IRDataset(csv_path=csv_path, image_root=image_root, bbox_root=bbox_root, bbox_pattern=bbox_pattern)
    test_dataset = IRDataset(csv_path=csv_path, image_root=image_root, bbox_root=bbox_root, bbox_pattern=bbox_pattern)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #TODO Shuffle을 켜야하나?
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = AuxDetScratch(meta_in_dim=train_dataset.meta_dim, meta_hidden=64, meta_out_dim=128, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(train_dataloader, model, optimizer, device)
        test_loss, test_acc = test_step(test_dataloader, model, device)
        print(f"Epoch {epoch} | Train Step : train_loss : {train_loss}, train_acc: {train_acc} | Test Step : test_loss : {test_loss}, test_acc : {test_acc}")

        torch.save(model.state_dict(), f"model_epoch_{epoch + 1:03d}.pt")
        print(f"saved: model_epoch_{epoch + 1:03d}.pt")

if __name__=="__main__":
    main()