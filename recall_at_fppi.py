"""
Compute Recall @ FPPI (False Positives Per Image) = 0.05, overall and
per-class ("sports-wise"), from a PaddleDetection eval run.

Requires:
    - bbox.json      (predictions, written by tools/eval.py)
    - ground truth COCO json (e.g. dataset/balls/test/annotations.json)

Usage:
    python recall_at_fppi.py \
        --gt dataset/balls/test/annotations.json \
        --pred bbox.json \
        --fppi 0.05 \
        --iou 0.5
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO


def box_iou(a, b):
    # a, b: [x, y, w, h]
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def compute_recall_at_fppi(gt_coco, preds, cat_id, num_images, target_fppi, iou_thresh):
    """
    preds: list of dicts with image_id, bbox, score, category_id (already
           filtered to this cat_id)
    Returns (recall_at_fppi, score_threshold_used, actual_fppi_at_that_point)
    """
    # ground truth boxes per image for this category
    gt_by_image = defaultdict(list)
    total_gt = 0
    for ann_id in gt_coco.getAnnIds(catIds=[cat_id]):
        ann = gt_coco.anns[ann_id]
        gt_by_image[ann['image_id']].append(ann['bbox'])
        total_gt += 1

    if total_gt == 0:
        return None, None, None

    # sort predictions by score descending
    preds_sorted = sorted(preds, key=lambda p: -p['score'])

    matched_gt = defaultdict(set)  # image_id -> set of matched gt indices
    tp_count = 0
    fp_count = 0

    # walk down the score-sorted list, tracking recall and FPPI at each step
    best_recall = 0.0
    best_thresh = None
    best_fppi = None

    for p in preds_sorted:
        img_id = p['image_id']
        pred_box = p['bbox']
        gts = gt_by_image.get(img_id, [])

        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt_box in enumerate(gts):
            if idx in matched_gt[img_id]:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt[img_id].add(best_gt_idx)
            tp_count += 1
        else:
            fp_count += 1

        recall = tp_count / total_gt
        fppi = fp_count / num_images

        # track the point where fppi first reaches/exceeds target
        # (recall at the tightest threshold still under/at target fppi)
        if fppi <= target_fppi:
            best_recall = recall
            best_thresh = p['score']
            best_fppi = fppi

    return best_recall, best_thresh, best_fppi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--fppi', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.5)
    args = ap.parse_args()

    gt_coco = COCO(args.gt)
    preds_all = json.load(open(args.pred))
    num_images = len(gt_coco.getImgIds())

    cat_ids = gt_coco.getCatIds()
    cats = gt_coco.loadCats(cat_ids)

    print(f"Test set: {num_images} images")
    print(f"Target FPPI: {args.fppi}, IoU threshold: {args.iou}\n")

    print("=== Sports-wise (per-class) Recall @ FPPI={} ===".format(args.fppi))
    per_class_recall = {}
    for cat in cats:
        cat_id = cat['id']
        preds_this_class = [p for p in preds_all if p['category_id'] == cat_id]
        recall, thresh, actual_fppi = compute_recall_at_fppi(
            gt_coco, preds_this_class, cat_id, num_images, args.fppi, args.iou)
        if recall is None:
            print(f"  {cat['name']}: no ground truth instances found")
            continue
        per_class_recall[cat['name']] = recall
        print(f"  {cat['name']:15s} recall={recall:.3f}  "
              f"(score_thresh={thresh:.3f}, actual_fppi={actual_fppi:.4f})")

    print("\n=== Overall Recall @ FPPI={} (all classes combined) ===".format(args.fppi))
    recall, thresh, actual_fppi = compute_recall_at_fppi(
        gt_coco, preds_all, None, num_images, args.fppi, args.iou)
    # note: overall pooled across classes needs total_gt across all cats,
    # handled by not filtering cat_id inside compute (see below patch)
    total_gt_all = len(gt_coco.getAnnIds())
    # recompute pooled properly
    gt_by_image_all = defaultdict(list)
    for ann in gt_coco.loadAnns(gt_coco.getAnnIds()):
        gt_by_image_all[ann['image_id']].append(ann['bbox'])

    preds_sorted = sorted(preds_all, key=lambda p: -p['score'])
    matched = defaultdict(set)
    tp, fp = 0, 0
    best_recall, best_thresh, best_fppi = 0.0, None, None
    for p in preds_sorted:
        img_id = p['image_id']
        gts = gt_by_image_all.get(img_id, [])
        best_iou, best_idx = 0.0, -1
        for idx, gt_box in enumerate(gts):
            if idx in matched[img_id]:
                continue
            iou = box_iou(p['bbox'], gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_iou >= args.iou and best_idx >= 0:
            matched[img_id].add(best_idx)
            tp += 1
        else:
            fp += 1
        r = tp / total_gt_all
        f = fp / num_images
        if f <= args.fppi:
            best_recall, best_thresh, best_fppi = r, p['score'], f

    print(f"  Overall: recall={best_recall:.3f} "
          f"(score_thresh={best_thresh}, actual_fppi={best_fppi})")

    print("\nNote: 'recall at FPPI=X' means the recall achieved when using the")
    print("confidence threshold that keeps false-positives-per-image at or")
    print("below X. If actual_fppi < target, it means predictions never reached")
    print("that many false positives even at low thresholds (good sign), and")
    print("this is the recall at the lowest threshold tested.")


if __name__ == '__main__':
    main()
