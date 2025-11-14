import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead as TVRPNHead, RegionProposalNetwork
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads

class RPNHead(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        self.head = TVRPNHead(out_channels, num_anchors=3)
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            head=self.head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(training=2000, testing=1000),
            post_nms_top_n=dict(training=1000, testing=300),
            nms_thresh=0.7
        )

    def forward(self, images, features, targets=None):
        return self.rpn(images, features, targets)

class ROIHead(nn.Module):
    def __init__(self, featmap_names=("c1","c2","c3","c4","pool"), out_channels=256, num_classes=2):
        super().__init__()
        self.pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=7, sampling_ratio=2)
        self.box_head = TwoMLPHead(in_channels=out_channels * 7 * 7, representation_size=1024)
        self.box_predictor = FastRCNNPredictor(1024, num_classes)
        self.roi_heads = RoIHeads(
            box_roi_pool=self.pool,
            box_head=self.box_head,
            box_predictor=self.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )

    def forward(self, features, proposals, image_sizes, targets=None):
        return self.roi_heads(features, proposals, image_sizes, targets)
