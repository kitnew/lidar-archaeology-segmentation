# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

import sys

from src.models.DeepLab_V3 import create_model
from src.datasets.dem_dataset import DEMTilesDataset

IGNORE_INDEX = -1

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight, reduction="none", bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_w * bce + self.dice_w * dice

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
            valid = valid.unsqueeze(1)
            preds = preds * valid
            targets = targets * valid
            valid_mask = valid.bool()
        else:
            valid_mask = None
        
        # Flatten tensors and convert to numpy
        preds_flat = preds.view(-1).cpu().numpy()
        targets_flat = targets.view(-1).cpu().numpy()
        
        # Binarize targets
        binary_targets = (targets_flat > 0.8).astype(float)  # Using 0.5 threshold for targets
        binary_preds = (preds_flat > threshold).astype(float)
        
        # Calculate confusion matrix elements
        tp = ((binary_preds > 0) & (binary_targets > 0.8)).sum()
        tn = ((binary_preds == 0) & (binary_targets <= 0.8)).sum()
        fp = ((binary_preds > 0) & (binary_targets <= 0.8)).sum()
        fn = ((binary_preds == 0) & (binary_targets > 0.8)).sum()
        
        # Pixel-level metrics
        eps = 1e-10
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        # Class-wise metrics
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # IoU (Jaccard Index) for each class
        iou_positive = tp / (tp + fp + fn + eps)
        
        # Calculate MSE for confidence values
        mse = nn.MSELoss()(probs, targets).item()
        
        return {
            # Overall metrics
            'accuracy': accuracy,   
            'mse': mse,
            
            # Class-specific metrics
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou_positive,
            
            # Additional metrics
            'specificity': tn / (tn + fp + eps),  # True Negative Rate
        }

def train_epoch(train_loader, model, criterion, optimizer):
    
    model.train()
    epoch_loss = 0.0
    total_correct = 0
    total_pixels = 0

    pbar = tqdm(train_loader, desc="Training", file=sys.stdout)
    
    for index, batch in enumerate(pbar):
        try:
            images = batch["dem"].to(device)
            masks = batch["mask"].to(device)
            valid = batch["valid"].to(device).unsqueeze(1).float()

            if images.shape[0] < 2:
                continue
            
            pred = model(images)['out']

            loss = criterion(pred, masks.unsqueeze(1).float())

            loss = (loss * valid).sum() / valid.sum()

            # Calculate accuracy
            pred_masks = (torch.sigmoid(pred) > 0.5).float()
            correct = (pred_masks == masks.unsqueeze(1)).float()
            correct = (correct * valid).sum()
            total_correct += correct.item()
            total_pixels += valid.sum().item()

            accuracy = correct / valid.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy.item()})
            epoch_loss += loss.item()
        except Exception as e:
            print(f"Error processing batch {index}: {e}")
            continue
        
    avg_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    return epoch_loss / len(train_loader), avg_accuracy

def validate(val_loader, model, criterion):
    model.eval()
    epoch_loss = 0.0
    total_correct = 0
    total_pixels = 0
    
    metrics = {
        'accuracy': [],   
        'mse': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'specificity': []
    }
    
    pbar = tqdm(val_loader, desc="Validation", file=sys.stdout)
    
    with torch.no_grad():
        for batch in pbar:
            images = batch["dem"].to(device)
            masks = batch["mask"].to(device)
            valid = batch["valid"].to(device).unsqueeze(1).float()
            
            pred = model(images)['out']
            
            # Calculate metrics
            batch_metrics = calculate_metrics(pred, masks, valid)
            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])
            
            loss = criterion(pred, masks.unsqueeze(1).float())
            
            loss = (loss * valid).sum() / valid.sum()
            
            # Calculate accuracy
            pred_masks = (torch.sigmoid(pred) > 0.8).float()
            correct = (pred_masks == masks.unsqueeze(1)).float()
            correct = (correct * valid).sum()
            total_correct += correct.item()
            total_pixels += valid.sum().item()
            
            accuracy = correct / valid.sum()
            
            pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy.item(), "IoU": np.mean(metrics['iou'])})
            epoch_loss += loss.item()
    avg_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    return epoch_loss / len(val_loader), avg_accuracy, avg_metrics

def train_loop(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, save_path):
    
    best_val_loss = float('inf')
    print("-" * 50)
    print("*" * 16 + " Training started " + "*" * 16)
    print("-" * 50)
    
    for epoch in range(num_epochs):
        print("#" * 20, f"Epoch {epoch+1}/{num_epochs}", "#" * 20)
        train_loss, train_accuracy = train_epoch(train_dataloader, model, criterion, optimizer)
        val_loss, val_accuracy, val_metrics = validate(val_dataloader, model, criterion)
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_accuracy:.4f} | Val Accuracy: {val_accuracy:.4f}')
        print("#### Metrics ####")
        for key, value in val_metrics.items():
            print(f'{key}: {value:.4f}')
        print("#"*17)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")
        print("")
    
    print("-" * 50)
    print("*" * 15 + " Training finished " + "*" * 15)
    print("-" * 50)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeepLabV3 model on DEM tiles.')
    parser.add_argument('--model', '-m', type=str, choices=['DeepLabV3'], default='DeepLabV3', help='Model name')
    parser.add_argument('--backbone', '-bone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone name')
    parser.add_argument('--pretrained', '-p', type=bool, default=False, help='Use pretrained weights')
    parser.add_argument('--dataset', '-d', type=str, default='dem_dataset', choices=['dem_dataset'], help='Dataset name')
    parser.add_argument('--tile-size', '-t', type=int, default=64, help='Tile size')
    parser.add_argument('--stride', '-s', type=int, default=32, help='Stride')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=4, help='Number of workers')
    parser.add_argument('--num-epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=0, help='Weight decay')
    parser.add_argument('--bce-weight', '-bw', type=float, default=0.5, help='BCEWithLogitsLoss weight')
    parser.add_argument('--dice-weight', '-dw', type=float, default=0.5, help='DiceLoss weight')
    parser.add_argument('--pos-weight', '-pw', type=float, default=20.62, help='Positive weight for BCEWithLogitsLoss')
    parser.add_argument('--reduction', type=str, default='none', choices=['none', 'mean', 'sum'], help='Reduction for BCEWithLogitsLoss')
    parser.add_argument('--save-path', type=str, default=False, help="Model save path")
        
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    match(args.model):
        case 'DeepLabV3':
            model = create_model(backbone=args.backbone, pretrained=args.pretrained, eval=False).to(device)
        case _:
            raise ValueError(f"Unknown model: {args.model}")
    
    match(args.dataset):
        case 'dem_dataset':
            data_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/"
            dem_train_path = data_path + "DEM0.80(5)_normalized_train.npz"
            dem_val_path = data_path + "DEM0.80(5)_normalized_val.npz"
            mask_train_path = data_path + "mounds_mask0.80(5)_shadowed_train.npy"
            mask_val_path = data_path + "mounds_mask0.80(5)_shadowed_val.npy"
            
            
            train_dataset = DEMTilesDataset(
                dem_path=dem_train_path,
                mask_path=mask_train_path,
                tile_size=args.tile_size,
                stride=args.stride,
                pos_only=True,
                transforms=True
            )
            
            val_dataset = DEMTilesDataset(
                dem_path=dem_val_path,
                mask_path=mask_val_path,
                tile_size=args.tile_size,
                stride=args.stride,
                pos_only=True,
                transforms=False
            )

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
            
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
    if not args.save_path:
        from datetime import datetime
        now = datetime.now()
        args.save_path = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_model.pth"
    
    # Initialize loss function and optimizer
    criterion = BCEDiceLoss(pos_weight=torch.tensor([args.pos_weight], device=device), reduction=args.reduction, bce_weight=args.bce_weight, dice_weight=args.dice_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    print("-" * 50)
    print("*" * 20 + " Parameters " + "*" * 20)
    print("")
    print("Using model: ", args.model)
    print("Using backbone: ", args.backbone, "(pretrained: ", args.pretrained, ")")
    print("Using dataset: ", args.dataset)
    print("Using tile size: ", args.tile_size)
    print("Dataset tiles: ", len(train_dataloader.dataset))
    print("Using stride: ", args.stride)
    print("Using batch size: ", args.batch_size)
    print("Using num workers: ", args.num_workers)
    print("Using num epochs: ", args.num_epochs)
    print("Using learning rate: ", args.learning_rate)
    print("Using positive weight: ", args.pos_weight)
    print("Using reduction: ", args.reduction)
    print("-" * 50)
    print("\n" * 2)
    
    # Train with validation
    train_loop(
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        args.num_epochs,
        args.save_path,
    )