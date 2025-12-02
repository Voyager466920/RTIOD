import torch
from torch.utils.data import DataLoader

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:,None] + area2 - inter
    return inter / (union + 1e-8)

def average_precision(tp, fp, n_gt):
    tp = torch.tensor(tp).cumsum(0)
    fp = torch.tensor(fp).cumsum(0)
    recall = tp / max(1, n_gt)
    precision = tp / torch.clamp(tp + fp, min=1)
    mrec = torch.cat([torch.tensor([0.]), recall, torch.tensor([1.])])
    mpre = torch.cat([torch.tensor([0.]), precision, torch.tensor([0.])])
    for i in range(mpre.numel()-1, 0, -1):
        mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])
    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).flatten()
    ap = float(torch.sum((mrec[idx+1]-mrec[idx]) * mpre[idx+1]))
    return ap

@torch.inference_mode()
def eval_map(dataloader: DataLoader, model, device, iou_ths=(0.5,), return_per_class=False):
    model.eval()
    preds_by_cls = {}
    gts_by_cls = {}
    img_offset = 0
    for images, metas, targets in dataloader:
        images = [im.to(device) for im in images]
        metas = metas.to(device)
        detections = model(images, metas, targets=None)
        for j, det in enumerate(detections):
            img_id = img_offset + j
            gt = targets[j]
            for c in gt["labels"].tolist():
                gts_by_cls.setdefault(c, {})
            for c in torch.unique(gt["labels"]).tolist():
                gts_by_cls[c].setdefault(img_id, [])
            for b, c in zip(gt["boxes"], gt["labels"]):
                gts_by_cls[c.item()][img_id].append(b.cpu())
            boxes = det["boxes"].cpu()
            labels = det["labels"].cpu()
            scores = det["scores"].cpu()
            for b, c, s in zip(boxes, labels, scores):
                preds_by_cls.setdefault(c.item(), []).append((img_id, float(s), b))
        img_offset += len(images)

    classes = sorted(set(list(preds_by_cls.keys()) + list(gts_by_cls.keys())))
    results = {}
    per_class = {}

    coco_ious = [0.5 + 0.05 * i for i in range(10)]
    need_coco = (len(iou_ths) == 1 and abs(iou_ths[0] - 0.5) < 1e-6)
    all_ious = sorted(set(list(iou_ths) + (coco_ious if need_coco else [])))

    for c in classes:
        per_class[c] = {}

    class_gt_counts = {}
    for c in classes:
        gts = gts_by_cls.get(c, {})
        class_gt_counts[c] = sum(len(v) for v in gts.values())

    for iou_thr in all_ious:
        aps = []
        weights = []
        for c in classes:
            preds = preds_by_cls.get(c, [])
            gts = gts_by_cls.get(c, {})
            n_gt = sum(len(v) for v in gts.values())
            if n_gt == 0:
                per_class[c][f"mAP@{iou_thr:.2f}"] = 0.0
                continue
            preds.sort(key=lambda x: x[1], reverse=True)
            matched = {img_id: torch.zeros(len(gts[img_id]), dtype=torch.bool) for img_id in gts.keys()}
            tp, fp = [], []
            for img_id, score, box in preds:
                if img_id not in gts:
                    fp.append(1)
                    tp.append(0)
                    continue
                gt_boxes = torch.stack(gts[img_id]) if len(gts[img_id]) else torch.zeros((0, 4))
                ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0) if gt_boxes.numel() else torch.zeros(0)
                if len(ious) == 0:
                    fp.append(1)
                    tp.append(0)
                    continue
                best_iou, best_idx = torch.max(ious, dim=0)
                if best_iou >= iou_thr and not matched[img_id][best_idx]:
                    matched[img_id][best_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    fp.append(1)
                    tp.append(0)
            ap = average_precision(tp, fp, n_gt)
            aps.append(ap)
            weights.append(n_gt)
            per_class[c][f"mAP@{iou_thr:.2f}"] = ap
        if aps:
            results[f"mAP@{iou_thr:.2f}"] = float(sum(aps) / len(aps))
            if weights:
                w_sum = float(sum(weights))
                num = 0.0
                for ap, w in zip(aps, weights):
                    num += ap * w
                results[f"w-mAP@{iou_thr:.2f}"] = float(num / w_sum)
            else:
                results[f"w-mAP@{iou_thr:.2f}"] = 0.0
        else:
            results[f"mAP@{iou_thr:.2f}"] = 0.0
            results[f"w-mAP@{iou_thr:.2f}"] = 0.0

    if need_coco:
        vals = []
        for thr in coco_ious:
            key = f"mAP@{thr:.2f}"
            if key in results:
                vals.append(results[key])
        results["mAP@[0.50:0.95]"] = sum(vals) / len(vals) if vals else 0.0
        for c in classes:
            vals_c = []
            for thr in coco_ious:
                key = f"mAP@{thr:.2f}"
                if key in per_class[c]:
                    vals_c.append(per_class[c][key])
            per_class[c]["mAP@[0.50:0.95]"] = sum(vals_c) / len(vals_c) if vals_c else 0.0

        total_gt = sum(v for v in class_gt_counts.values() if v > 0)
        if total_gt > 0:
            num = 0.0
            for c in classes:
                w = class_gt_counts[c]
                if w <= 0:
                    continue
                ap_c = per_class[c].get("mAP@[0.50:0.95]", 0.0)
                num += ap_c * w
            results["w-mAP@[0.50:0.95]"] = float(num / total_gt)
        else:
            results["w-mAP@[0.50:0.95]"] = 0.0

    if return_per_class:
        return results, per_class
    return results
