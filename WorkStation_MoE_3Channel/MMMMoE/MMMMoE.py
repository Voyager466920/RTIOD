import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList

from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

from WorkStation_MoE.MMMMoE.Original_MMMMoE.MoEBlock import MoEBlock
from WorkStation_MoE_3Channel.MMMMoE.Backbone import Scratch, ImageExtractorResnet18, ImageExtractorResnet34, ImageExtractorResnet50
from WorkStation_MoE_3Channel.MMMMoE.RPN_ROI_Head import build_rpn_and_roi_heads


class MMMMoE_Detector(nn.Module):
    def __init__(self, num_classes:int=5, fpn_out:int=256, meta_dim:int=12, backbone="scratch"):
        super().__init__()
        self.backbone = MMMMoE(fpn_out=fpn_out, meta_dim=meta_dim, back_bone= backbone)
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
    def __init__(self, back_bone="scratch", fpn_out:int=256, num_experts=6, meta_dim=9, pretrained=True):
        super().__init__()
        self.backbone, backbone_out_channels = self._select_backbone(back_bone, pretrained)
        self.moeblock = MoEBlock(in_channels=backbone_out_channels[-1],hidden_channels=backbone_out_channels[-1],num_experts=num_experts,meta_dim=meta_dim)
        self.fpn = FeaturePyramidNetwork(in_channels_list=backbone_out_channels,out_channels=fpn_out)
        self.out_channels = fpn_out

    def _select_backbone(self, name, pretrained):
        if name == "scratch":
            return Scratch(), [128, 256, 512, 512]
        elif name == "resnet18":
            return ImageExtractorResnet18(pretrained=pretrained), [128, 256, 512, 512]
        elif name == "resnet34":
            return ImageExtractorResnet34(pretrained=pretrained), [128, 256, 512, 512]
        elif name == "resnet50":
            return ImageExtractorResnet50(pretrained=pretrained), [512, 1024, 2048, 2048]

        raise ValueError(f"invalid backbone: {name}")

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
