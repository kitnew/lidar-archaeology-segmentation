import torch
import torch.nn as nn
import numpy as np

def calculate_metrics(outputs, targets, valid=None, threshold=0.8):
    """Calculate metrics for binary semantic segmentation.
    
    Args:
        outputs: Model output logits (before sigmoid) of shape (B, 1, H, W)
        targets: Ground truth masks of shape (B, 1, H, W)
        valid: Optional mask indicating valid pixels
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary containing various segmentation metrics
    """
    with torch.no_grad():
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        
        # Create binary predictions
        preds = (probs > threshold).float()
        
        # Apply valid mask if provided
        if valid is not None:
            if valid.dim() == 3:
                valid = valid.unsqueeze(1)
            preds = preds * valid
            targets = targets * valid
        
        # Flatten tensors and convert to numpy
        preds_flat = preds.view(-1).cpu().numpy()
        targets_flat = targets.view(-1).cpu().numpy()
        
        # Binarize targets
        binary_targets = (targets_flat > 0.8).astype(float)
        binary_preds = (preds_flat > threshold).astype(float)
        
        # Calculate confusion matrix elements
        tp = ((binary_preds > 0) & (binary_targets > 0)).sum()
        tn = ((binary_preds == 0) & (binary_targets == 0)).sum()
        fp = ((binary_preds > 0) & (binary_targets == 0)).sum()
        fn = ((binary_preds == 0) & (binary_targets > 0)).sum()
        
        # Pixel-level metrics
        eps = 1e-10
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        # Class-wise metrics
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # IoU (Jaccard Index)
        iou_positive = tp / (tp + fp + fn + eps)
        
        # Calculate MSE for confidence values
        mse = nn.MSELoss()(probs, targets).item()
        
        return {
            'accuracy': accuracy,
            'mse': mse,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou_positive,
            'specificity': tn / (tn + fp + eps),  # True Negative Rate
        }
