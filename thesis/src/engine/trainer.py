import torch
import numpy as np
import sys
import logging
import os
from tqdm import tqdm

from utils.metrics import optimal_threshold, get_stats_counts

log = logging.getLogger(__name__)

def calculate_metrics_from_counts(tp, fp, fn, tn, eps=1e-10):
    iou_pos = tp / (tp + fp + fn + eps)
    iou_neg = tn / (tn + fp + fn + eps)
    iou = (iou_pos + iou_neg) / 2
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    mor10r = tp / (tp + fn + 10 * fp + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou_pos": iou_pos,
        "iou_neg": iou_neg,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mor10r": mor10r,
        "specificity": specificity,
        "accuracy": accuracy,
    }

class Trainer:
    def __init__(self, model, optimizer, criterion, device, threshold=0.5, logger=None, checkpoint_metric="iou_pos"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.threshold = threshold
        self.logger = logger
        self.checkpoint_metric = checkpoint_metric

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        epoch_loss = 0.0
        processed_batches = 0

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", file=sys.stdout)

        for batch in pbar:
            images = batch["data"].to(self.device)
            masks = batch["mask"].to(self.device)
            valid = batch["valid"].to(self.device)

            outputs = self.model(images)
            pred = outputs["out"] if isinstance(outputs, dict) else outputs

            loss = self.criterion(pred, masks, valid=valid)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            processed_batches += 1

            pbar.set_postfix({"loss": loss.item()})

            if self.logger:
                self.logger.log({"train/iter_loss": loss.item()})

        avg_loss = epoch_loss / max(processed_batches, 1)
        return avg_loss

    def validate(self, dataloader, epoch):
        self.model.eval()
        epoch_loss = 0.0
        processed_batches = 0

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}", file=sys.stdout)

        with torch.no_grad():
            for batch in pbar:
                images = batch["data"].to(self.device)
                masks = batch["mask"].to(self.device)
                valid = batch["valid"].to(self.device)

                outputs = self.model(images)
                pred = outputs["out"] if isinstance(outputs, dict) else outputs

                loss = self.criterion(pred, masks, valid=valid)
                epoch_loss += loss.item()
                processed_batches += 1

                probs = torch.sigmoid(pred)
                pred_bin = (probs > self.threshold).cpu().numpy().astype(bool)
                gt_bin = masks.cpu().numpy().astype(np.float32)
                valid_bin = valid.cpu().numpy().astype(bool)

                if valid_bin.sum() == 0:
                    pbar.set_postfix({"loss": loss.item(), "IoU positive": 0.0})
                    continue

                tp, fp, fn, tn = get_stats_counts(pred_bin, gt_bin, valid=valid_bin)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                running_iou = total_tp / (total_tp + total_fp + total_fn + 1e-10)
                pbar.set_postfix({"loss": loss.item(), "IoU positive": running_iou})

        avg_loss = epoch_loss / max(processed_batches, 1)
        avg_metrics = calculate_metrics_from_counts(total_tp, total_fp, total_fn, total_tn)

        best_t, _ = optimal_threshold(probs.cpu().numpy(), gt_bin, valid=valid_bin)
        self.threshold = best_t
        log.info(f"Threshold: {self.threshold}")

        return avg_loss, avg_metrics, (total_tp, total_fp, total_fn, total_tn), best_t

    def fit(self, train_loader, val_loader, epochs, save_dir):
        self.model.train()

        best_val_loss = float("inf")
        best_val_iou = float("-inf")
        best_val_f1 = float("-inf")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            log.info(f"--- Epoch {epoch}/{epochs} ---")

            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics, val_counts, val_threshold = self.validate(val_loader, epoch)
            total_tp, total_fp, total_fn, total_tn = val_counts

            log.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            log.info(f"Val Counts: TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")

            log_dict = {
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "epoch": epoch,
                "val/threshold": val_threshold,
            }

            for k, v in val_metrics.items():
                log.info(f"Val {k}: {v:.4f}")
                log_dict[f"val/{k}"] = v

            if self.logger:
                self.logger.log(log_dict)

            if val_metrics["iou_pos"] > best_val_iou if self.checkpoint_metric == "iou_pos" else val_metrics["f1"] > best_val_f1:
                best_val_iou = val_metrics["iou_pos"]
                best_val_f1 = val_metrics["f1"]
                ckpt_path = os.path.join(save_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_iou_pos": best_val_iou,
                        "val_f1": best_val_f1,
                        "threshold": self.threshold,
                    },
                    ckpt_path,
                )
                log.info(f"Saved new best model to {ckpt_path}")