import torch
from torch.utils.data import DataLoader

def box_iou(boxes1, boxes2):
    # boxes: [N,4] xyxy
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])      # [N,M,2]
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])      # [N,M,2]
    wh = (rb - lt).clamp(min=0)                          # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]                        # [N,M]
    union = area1[:,None] + area2 - inter
    return inter / (union + 1e-8)

def average_precision(tp, fp, n_gt):
    # tp/fp: list of 0/1 along score-desc order
    tp = torch.tensor(tp).cumsum(0)
    fp = torch.tensor(fp).cumsum(0)
    recall = tp / max(1, n_gt)
    precision = tp / torch.clamp(tp + fp, min=1)
    # 11-point interpolation not used; integrate precision envelope
    mrec = torch.cat([torch.tensor([0.]), recall, torch.tensor([1.])])
    mpre = torch.cat([torch.tensor([0.]), precision, torch.tensor([0.])])
    for i in range(mpre.numel()-1, 0, -1):
        mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).flatten()
    ap = float(torch.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))
    return ap

@torch.inference_mode()
def eval_map(dataloader: DataLoader, model, device, iou_ths=(0.5,)):
    model.eval()
    # 수집: 클래스별 예측/GT
    preds_by_cls = {}  # c -> list of (img_id, score, box[4])
    gts_by_cls = {}    # c -> dict img_id -> list of boxes
    img_offset = 0
    for images, metas, targets in dataloader:
        images = [im.to(device) for im in images]
        metas = metas.to(device)
        detections = model(images, metas, targets=None)
        for j, det in enumerate(detections):
            img_id = img_offset + j
            # GT 저장
            gt = targets[j]
            for c in gt["labels"].tolist():
                gts_by_cls.setdefault(c, {})
            for c in torch.unique(gt["labels"]).tolist():
                gts_by_cls[c].setdefault(img_id, [])
            for b, c in zip(gt["boxes"], gt["labels"]):
                gts_by_cls[c.item()][img_id].append(b.cpu())
            # 예측 저장
            boxes = det["boxes"].cpu()
            labels = det["labels"].cpu()
            scores = det["scores"].cpu()
            for b, c, s in zip(boxes, labels, scores):
                preds_by_cls.setdefault(c.item(), []).append((img_id, float(s), b))
        img_offset += len(images)

    # AP 계산
    classes = sorted(set(list(preds_by_cls.keys()) + list(gts_by_cls.keys())))
    results = {}
    for iou_thr in iou_ths:
        aps = []
        for c in classes:
            preds = preds_by_cls.get(c, [])
            gts = gts_by_cls.get(c, {})
            n_gt = sum(len(v) for v in gts.values())
            if n_gt == 0:
                continue
            preds.sort(key=lambda x: x[1], reverse=True)  # score desc
            matched = {img_id: torch.zeros(len(gts[img_id]), dtype=torch.bool) for img_id in gts.keys()}
            tp, fp = [], []
            for img_id, score, box in preds:
                if img_id not in gts:
                    fp.append(1); tp.append(0); continue
                gt_boxes = torch.stack(gts[img_id]) if len(gts[img_id]) else torch.zeros((0,4))
                ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0) if gt_boxes.numel() else torch.zeros(0)
                if len(ious)==0:
                    fp.append(1); tp.append(0); continue
                best_iou, best_idx = torch.max(ious, dim=0)
                if best_iou >= iou_thr and not matched[img_id][best_idx]:
                    matched[img_id][best_idx] = True
                    tp.append(1); fp.append(0)
                else:
                    fp.append(1); tp.append(0)
            ap = average_precision(tp, fp, n_gt)
            aps.append(ap)
        results[f"mAP@{iou_thr:.2f}"] = sum(aps)/len(aps) if aps else 0.0

    # COCO식 0.50:0.95
    if len(iou_ths) == 1 and abs(iou_ths[0]-0.5) < 1e-6:
        ious = [0.5 + 0.05*i for i in range(10)]
        coco = eval_map(dataloader, model, device, tuple(ious))
        results["mAP@[0.50:0.95]"] = sum(v for k,v in coco.items())/len(coco) if coco else 0.0
    return results
