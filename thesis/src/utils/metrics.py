import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import sys

@dataclass
class Metrics:
    iou_pos: float
    iou_neg: float
    iou: float
    precision: float
    recall: float
    f1_score: float
    mor10r: float
    specificity: float
    accuracy: float

def get_stats_counts(pred_bin, gt, valid=None):
    pred_bin = np.asarray(pred_bin, dtype=np.bool_).reshape(-1)
    gt = np.asarray(gt, dtype=np.bool_).reshape(-1)

    if valid is not None:
        valid = np.asarray(valid, dtype=np.bool_).reshape(-1)
        pred_bin = pred_bin[valid]
        gt = gt[valid]

    tp = np.sum(pred_bin & gt)
    fp = np.sum(pred_bin & ~gt)
    fn = np.sum(~pred_bin & gt)
    tn = np.sum(~pred_bin & ~gt)

    return int(tp), int(fp), int(fn), int(tn)
    
def compute_metrics(tp, fp, fn, tn):
    eps = 1e-10

    iou_pos = tp / (tp + fp + fn + eps)
    iou_neg = tn / (tn + fp + fn + eps)
    iou = (iou_pos + iou_neg) / 2
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    mor10r = tp / (tp + fn + 10 * fp + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return Metrics(iou_pos, iou_neg, iou, precision, recall, f1, mor10r, specificity, accuracy)

def optimal_threshold(probs, targets, valid=None, metric='iou_pos', default_t=0.5):
    best_iou_pos, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity = 0, 0, 0, 0, 0, 0, 0
    thresholds = np.linspace(0.2, 0.99, 100)
    best_t = default_t

    pbar = tqdm(thresholds, desc="Optimizing", file=sys.stdout)

    best_metrics = compute_metrics(0, 0, 0, 0)

    for t in pbar:
        pred_bin = (probs > t).astype(bool)
        tp, fp, fn, tn = get_stats_counts(pred_bin, targets, valid)
        metrics = compute_metrics(tp, fp, fn, tn)
        if metric == 'iou_pos':
            if metrics.iou_pos > best_iou_pos:
                best_iou_pos = metrics.iou_pos
                best_t, best_metrics = t, metrics
        elif metric == 'f1':
            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                best_t, best_metrics = t, metrics
        elif metric == 'mor10r':
            if metrics.mor10r > best_mor10r:
                best_mor10r = metrics.mor10r
                best_t, best_metrics = t, metrics
        elif metric == 'specificity':
            if metrics.specificity > best_specificity:
                best_specificity = metrics.specificity
                best_t, best_metrics = t, metrics
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        pbar.set_postfix({"t": f"{t:.3f}", "best": f"{getattr(best_metrics, metric):.4f}"})

    return best_t, best_metrics