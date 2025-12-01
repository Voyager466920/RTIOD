from typing import Tuple
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


def build_rpn_and_roi_heads(
    backbone_out_channels: int,
    num_classes: int,
    featmap_names: Tuple[str, ...] = ("0", "1", "2", "3"),
):

    num_levels = len(featmap_names)

    sizes = tuple((32 * (2 ** i),) for i in range(num_levels))
    aspect_ratios = ((0.5, 1.0, 2.0),) * num_levels

    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios,
    )

    rpn_head = RPNHead(
        in_channels=backbone_out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
    )

    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": 2000, "testing": 1000},
        post_nms_top_n={"training": 1000, "testing": 300},
        nms_thresh=0.7,
    )

    resolution = 7
    box_head = TwoMLPHead(
        backbone_out_channels * resolution * resolution,
        1024,
    )

    box_predictor = FastRCNNPredictor(
        in_channels=1024,
        num_classes=num_classes,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=featmap_names,
        output_size=7,
        sampling_ratio=2,
    )

    roi_heads = RoIHeads(
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=512,
        positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )

    return rpn, roi_heads