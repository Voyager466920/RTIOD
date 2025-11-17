import torch
from torch.utils.data import DataLoader


def main():
    epochs = 10

    test_dataset = Dataset()
    train_dataset = Dataset()

    test_dataloader = DataLoader(test_dataset)
    train_datloader = DataLoader(train_dataset)

    for epoch in range(epochs):
        train_loss, train_acc = train_step()
        test_loss, test_acc = test_step()