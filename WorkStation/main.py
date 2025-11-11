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

    train_dataset = IRDataset()
    test_dataset = IRDataset()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #TODO Shuffle을 켜야하나?
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = AuxDetScratch(meta_in_dim=9, meta_out_dim= 10, meta_hidden=32)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss, train_acc = train_step(train_dataloader, model, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(test_dataloader, model, loss_fn, device)
        print(f"Train Step : train_loss : {train_loss}, train_acc: {train_acc} | Test Step : test_loss : {test_loss}, test_acc : {test_acc}")
