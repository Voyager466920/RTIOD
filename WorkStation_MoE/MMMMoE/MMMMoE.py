import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList

from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from WorkStation_MoE.MMMMoE.MoEBlock import MoEBlock
from WorkStation_MoE.MMMMoE.RPN_ROI_Head import build_rpn_and_roi_heads


class MMMMoE_Detector(nn.Module):
    def __init__(self, num_classes:int=5, fpn_out:int=256):
        super().__init__()
        self.backbone = MMMMoE(fpn_out=fpn_out)
        self.rpn, self.roi_heads = build_rpn_and_roi_heads(
            backbone_out_channels=self.backbone.out_channels,
            num_classes=num_classes,
            featmap_names=("0", "1", "2", "3"),
        )

    def _to_image_list(self, images):
        if isinstance(images, torch.Tensor):
            b, c, h, w = images.shape
            image_sizes = [(h, w)] * b
            return ImageList(images, image_sizes)
        else:
            sizes = [img.shape[-2:] for img in images]
            images = torch.stack(images, dim=0)
            return ImageList(images, sizes)

    def forward(self, images, meta, targets=None):
        images_list = self._to_image_list(images)
        features, balance_loss = self.backbone(images_list.tensors, meta)

        proposals, rpn_losses = self.rpn(images_list, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images_list.image_sizes, targets
        )

        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            losses["moe_balance_loss"] = balance_loss
            return losses

        return detections



class MMMMoE(nn.Module):
    def __init__(self, fpn_out:int=256, num_experts=6, meta_dim=9):
        super().__init__()
        self.backbone = BackBone()
        self.moeblock = MoEBlock(num_experts=num_experts)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 512],
            out_channels=fpn_out
        )
        self.out_channels = fpn_out

    def forward(self, images, meta):
        c1, c2, c3, c4 = self.backbone(images)
        moe_c4, balance_loss = self.moeblock(c4, meta)

        feats = OrderedDict()
        feats["0"] = c2
        feats["1"] = c3
        feats["2"] = moe_c4
        feats["3"] = F.max_pool2d(moe_c4, 2)

        fpn_feats = self.fpn(feats)
        return fpn_feats, balance_loss


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BasicBlock(128)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            BasicBlock(256)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            BasicBlock(512)
        )

    def forward(self, x):
        c1 = self.stage1(x)   # 192x144
        c2 = self.stage2(c1)   # 96x72
        c3 = self.stage3(c2)   # 48x36
        c4 = self.stage4(c3)   # 24x18
        return c1, c2, c3, c4

class BasicBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        r = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + r)
        return x