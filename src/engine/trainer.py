import torch
import numpy as np
import sys
import logging
import os
from tqdm import tqdm
#from .metrics import calculate_metrics
from dataclasses import dataclass

log = logging.getLogger(__name__)

def compute_metrics(pred_bin, gt, eps=1e-10):
    """
    Compute segmentation metrics using numpy arrays.
    pred_bin and gt should be boolean numpy arrays.
    """
    tp = np.sum(pred_bin & gt)
    fp = np.sum(pred_bin & ~gt)
    fn = np.sum(~pred_bin & gt)
    tn = np.sum(~pred_bin & ~gt)

    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    mor10r = tp / (tp + fn + 10 * fp + eps)
    specificity = tn / (tn + fp + eps)

    return {
        "iou": iou,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mor10r": mor10r,
        "specificity": specificity
    }


class Trainer:
    def __init__(self, model, optimizer, criterion, device, threshold=0.8, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.threshold = threshold
        self.logger = logger # Could be wandb, tensorboard, or none

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_pixels = 0

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", file=sys.stdout)
        
        for batch in pbar:
            images = batch["data"].to(self.device)
            masks = batch["mask"].to(self.device)
            valid = batch["valid"].to(self.device).unsqueeze(1).float() if "valid" in batch else torch.ones_like(masks)

            if images.shape[0] < 2:
                continue
            
            # Forward pass
            outputs = self.model(images)
            pred = outputs['out'] if isinstance(outputs, dict) else outputs

            # Ensure mask dims match predictions
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()

            import inspect
            if hasattr(self.criterion, "forward") and "valid" in inspect.signature(self.criterion.forward).parameters:
                loss = self.criterion(pred, masks, valid=valid)
            else:
                loss = self.criterion(pred, masks)

            # Apply valid mask to loss if reduction is none
            if loss.dim() > 0:
                loss = (loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)

            # Accuracy calc
            pred_masks = (torch.sigmoid(pred) > 0.5).float()
            correct = (pred_masks == masks).float() * valid
            
            total_correct += correct.sum().item()
            total_pixels += valid.sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iter_acc = correct.sum().item() / max(valid.sum().item(), 1.0)
            pbar.set_postfix({"loss": loss.item(), "acc": iter_acc})
            epoch_loss += loss.item()

            if self.logger:
                self.logger.log({"train/iter_loss": loss.item(), "train/iter_acc": iter_acc})

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = total_correct / max(total_pixels, 1)
        
        return avg_loss, avg_acc

    def validate(self, dataloader, epoch):
        self.model.eval()
        epoch_loss = 0.0
        total_correct = 0
        total_pixels = 0
        
        metrics_history = {k: [] for k in ['iou', 'f1', 'precision', 'recall', 'accuracy', 'mor10r', 'specificity']}
        
        pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}", file=sys.stdout)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch["data"].to(self.device)
                masks = batch["mask"].to(self.device)
                valid = batch["valid"].to(self.device).unsqueeze(1).float() if "valid" in batch else torch.ones_like(masks)
                
                outputs = self.model(images)
                pred = outputs['out'] if isinstance(outputs, dict) else outputs
                
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1).float()

                #if self.logger and batch_idx == 0: # логируем только первый батч для экономии места
                #    import wandb
                #    self.logger.log({
                #        "predictions": [wandb.Image(images[0].cpu(), caption="Input"), 
                #                        wandb.Image(torch.sigmoid(pred)[0].cpu(), caption="Prediction"),
                #                        wandb.Image(masks[0].cpu(), caption="Ground Truth")]
                #    })
                
                # Binarize for metrics
                pred_bin = (torch.sigmoid(pred) > self.threshold).cpu().numpy().astype(bool)
                
                # Metrics
                batch_metrics = compute_metrics(pred_bin, masks.cpu().numpy().astype(bool))
                for k in metrics_history:
                    if k in batch_metrics:
                        metrics_history[k].append(batch_metrics[k])
                
                # Loss
                import inspect
                if hasattr(self.criterion, "forward") and "valid" in inspect.signature(self.criterion.forward).parameters:
                    loss = self.criterion(pred, masks, valid=valid)
                else:
                    loss = self.criterion(pred, masks)
                if loss.dim() > 0:
                    loss = (loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)
                
                # Accuracy calc
                pred_masks = (torch.sigmoid(pred) > self.threshold).float()
                correct = (pred_masks == masks).float() * valid
                
                total_correct += correct.sum().item()
                total_pixels += valid.sum().item()
                
                iter_acc = correct.sum().item() / max(valid.sum().item(), 1)
                iter_iou = np.mean(metrics_history['iou']) if len(metrics_history['iou']) > 0 else 0
                
                pbar.set_postfix({"loss": loss.item(), "acc": iter_acc, "IoU": iter_iou})
                epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = total_correct / max(total_pixels, 1)
        avg_metrics = {k: np.mean(v) for k, v in metrics_history.items() if len(v) > 0}
        
        return avg_loss, avg_acc, avg_metrics

    def fit(self, train_loader, val_loader, epochs, save_dir):
        best_val_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            log.info(f"--- Epoch {epoch}/{epochs} ---")
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_metrics = self.validate(val_loader, epoch)
            
            log.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            log.info(f"Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
            
            log_dict = {
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "val/epoch_loss": val_loss,
                "val/epoch_acc": val_acc,
                "epoch": epoch
            }
            
            for k, v in val_metrics.items():
                log.info(f"Val {k}: {v:.4f}")
                log_dict[f"val/{k}"] = v
                
            if self.logger:
                self.logger.log(log_dict)
            
            # Simple checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(save_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, ckpt_path)
                log.info(f"Saved new best model to {ckpt_path}")
