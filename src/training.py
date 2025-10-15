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

def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    total_correct = 0
    total_pixels = 0

    pbar = tqdm(train_loader, desc="Training", file=sys.stdout)
    
    for batch in pbar:
        images = batch["dem"].to(device)
        masks = batch["mask"].to(device)
        valid = batch["valid"].to(device).unsqueeze(1).float()
        
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
        
    avg_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    return epoch_loss / len(train_loader), avg_accuracy

def validate(val_loader, model, criterion):
    model.eval()
    epoch_loss = 0.0
    total_correct = 0
    total_pixels = 0
    
    pbar = tqdm(val_loader, desc="Validation", file=sys.stdout)
    
    with torch.no_grad():
        for batch in pbar:
            images = batch["dem"].to(device)
            masks = batch["mask"].to(device)
            valid = batch["valid"].to(device).unsqueeze(1).float()
            
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
            
            pbar.set_postfix({"loss": loss.item(), "accuracy": accuracy.item()})
            epoch_loss += loss.item()
    avg_accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    return epoch_loss / len(val_loader), avg_accuracy

def train_loop(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, save_path):
    
    best_val_loss = float('inf')
    print("-" * 50)
    print("*" * 16 + " Training started " + "*" * 16)
    print("-" * 50)
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(train_dataloader, model, criterion, optimizer)
        val_loss, val_accuracy = validate(val_dataloader, model, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")
    
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
    parser.add_argument('--split-ratio', type=tuple, default=(0.7, 0.15, 0.15), help='Split ratio for train, val, test')
    parser.add_argument('--tile-size', '-t', type=int, default=64, help='Tile size')
    parser.add_argument('--stride', '-s', type=int, default=32, help='Stride')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=4, help='Number of workers')
    parser.add_argument('--num-epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save-path', type=str, default=False, help="Model save path")
        
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    match(args.model):
        case 'DeepLabV3':
            model = create_model(backbone=args.backbone, pretrained=args.pretrained, eval=False).to(device)
        case _:
            raise ValueError(f"Unknown model: {args.model}")
    
    print("Using model: ", args.model)
    print("Using backbone: ", args.backbone)
    print("Using dataset: ", args.dataset)
    print("Using tile size: ", args.tile_size)
    print("Using stride: ", args.stride)
    print("Using batch size: ", args.batch_size)
    print("Using num workers: ", args.num_workers)
    print("Using num epochs: ", args.num_epochs)
    print("Using learning rate: ", args.learning_rate)
    
    match(args.dataset):
        case 'dem_dataset':
            dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM_normalized.npz"
            mask_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/mounds_mask_shadowed.npy"
            
            dataset = DEMTilesDataset(
                dem_path=dem_path,
                mask_path=mask_path,
                tile_size=args.tile_size,
                stride=args.stride,
            )
            
            # Получаем все координаты
            coords = np.array(dataset.coords)  # shape: (N, 2) -> [y, x]

            # Разделим по оси Y (например, верх 70%, низ 30%)
            y_values = coords[:, 0]
            y_threshold = np.percentile(y_values, 70)

            train_indices = [i for i, (y, _) in enumerate(dataset.coords) if y < y_threshold]
            val_indices   = [i for i, (y, _) in enumerate(dataset.coords) if y >= y_threshold]

            # Создаём сабсеты
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)

            train_dataloader = DataLoader(train_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
    if not args.save_path:
        from datetime import datetime
        now = datetime.now()
        args.save_path = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_model.pth"
    
    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.62], device=device), reduction="mean").to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
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