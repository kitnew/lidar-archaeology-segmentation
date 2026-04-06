import numpy as np
from dataclasses import dataclass

@dataclass
class Metrics:
    iou: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mor10r: float
    specificity: float

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

    iou = tp / (tp + fp + fn + 1e-8)

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    mor10r = tp / (tp + fn + 10*fp + 1e-8)
    specificity = tn / (tn + fp + eps)

    return Metrics(iou, acc, precision, recall, f1, mor10r, specificity)

def optimal_threshold(probs, targets, valid=None, metric='iou', default_t=0.5):
    best_iou, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity = 0, 0, 0, 0, 0, 0, 0
    thresholds = np.linspace(0.5, 0.999, 100)
    best_t = default_t

    for t in thresholds:
        pred_bin = (probs > t).astype(bool)
        tp, fp, fn, tn = get_stats_counts(pred_bin, targets, valid)
        metrics = compute_metrics(tp, fp, fn, tn)
        if metric == 'iou':
            if metrics.iou > best_iou:
                best_t, best_iou, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity = t, metrics.iou, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score, metrics.mor10r, metrics.specificity
        elif metric == 'f1':
            if metrics.f1_score > best_f1:
                best_t, best_iou, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity = t, metrics.iou, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score, metrics.mor10r, metrics.specificity
        elif metric == 'mor10r':
            if metrics.mor10r > best_mor10r:
                best_t, best_iou, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity = t, metrics.iou, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score, metrics.mor10r, metrics.specificity
        elif metric == 'specificity':
            if metrics.specificity > best_specificity:
                best_t, best_iou, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity = t, metrics.iou, metrics.accuracy, metrics.precision, metrics.recall, metrics.f1_score, metrics.mor10r, metrics.specificity
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return best_t, Metrics(best_iou, best_acc, best_precision, best_recall, best_f1, best_mor10r, best_specificity)