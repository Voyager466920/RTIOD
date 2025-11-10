import torch
import torch.nn as nn

class M2DMScratch(nn.Module):
    def __init__(self, image_data, meta_data):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Linear(in_features=128, out_features=256)
        )

    def forward(self, x, a):
        temp1 = self.stage1(x)
        return x