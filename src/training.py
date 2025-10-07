# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.models.DeepLab_V3 import create_model
from src.datasets.dem_dataset import DEMTilesDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_accuracy(outputs, targets, ignore_index=255):
    """Calculate pixel accuracy, ignoring pixels with ignore_index."""
    with torch.no_grad():
        # Get predicted class (ignore ignore_index)
        _, preds = torch.max(outputs, 1)
        valid_pixels = (targets != ignore_index)
        correct = (preds[valid_pixels] == targets[valid_pixels]).sum().item()
        total = valid_pixels.sum().item()
        return correct / total if total > 0 else 0.0

def train_epoch(dataloader, model, criterion, optimizer):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # Прогресс-бар
    pbar = tqdm(dataloader, desc='Training')
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        
        # Calculate accuracy
        acc = calculate_accuracy(outputs, masks, ignore_index=255)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_acc += acc
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{acc*100:.1f}%'})
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def validate(dataloader, model, criterion):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            acc = calculate_accuracy(outputs, masks, ignore_index=255)
            
            val_loss += loss.item()
            val_acc += acc
            pbar.set_postfix({'val_loss': loss.item(), 'val_acc': f'{acc*100:.1f}%'})
    
    return val_loss / len(dataloader), val_acc / len(dataloader)

def train_loop(train_dataloader, val_dataloader, model, criterion, optimizer, num_epochs, save_path):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training
        train_loss, train_acc = train_epoch(train_dataloader, model, criterion, optimizer)
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        
        # Validation
        val_loss, val_acc = validate(val_dataloader, model, criterion)
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%')
        
        # Save best model (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
            }, save_path)
            print(f'Model saved to {save_path} (val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f}%)')
        
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeepLabV3 model on DEM tiles.')
    parser.add_argument('--mode', '-M', type=str, choices=['slurm', 'local'], default='slurm', help='Mode')
    parser.add_argument('--device', '-D', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device')
    parser.add_argument('--model', '-m', type=str, choices=['DeepLabV3'], default='DeepLabV3', help='Model name')
    parser.add_argument('--backbone', '-bone', type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='Backbone name')
    parser.add_argument('--pretrained', '-p', type=bool, default=False, help='Use pretrained weights')
    parser.add_argument('--dataset', '-d', type=str, default='dem_dataset', choices=['dem_dataset'], help='Dataset name')
    parser.add_argument('--split-ratio', type=tuple, default=(0.7, 0.15, 0.15), help='Split ratio for train, val, test')
    parser.add_argument('--tile-size', '-t', type=int, default=1024, help='Tile size')
    parser.add_argument('--stride', '-s', type=int, default=512, help='Stride')
    parser.add_argument('--ignore-index', '-i', type=int, default=255, help='Ignore index')
    parser.add_argument('--batch-size', '-b', type=int, default=2, help='Batch size')
    parser.add_argument('--num-workers', '-w', type=int, default=4, help='Number of workers')
    parser.add_argument('--num-epochs', '-e', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-path', type=str, default=False, help="Model save path")
    #parser.add_argument('--resume', '-r', type=bool, default=False, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    print(args)
    
    match(args.device):
        case 'cuda':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        case 'cpu':
            device = torch.device('cpu')
        case _:
            raise ValueError(f"Unknown device: {args.device}")
    print("Device: ", device)
        
    
    match(args.model):
        case 'DeepLabV3':
            model = create_model(backbone=args.backbone, pretrained=args.pretrained, eval=False).to(device)
        case _:
            raise ValueError(f"Unknown model: {args.model}")
        
    match(args.dataset):
        case 'dem_dataset':
            match(args.mode):
                case 'slurm':
                    dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/raw/DEM.npz"
                    mask_path = "/home/nc225mj/lidar-archaeology-segmentation/data/raw/Mounds_raster_mask_opened_closed.npy"
                case 'local':
                    dem_path = "/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/raw/DEM.npz"
                    mask_path = "/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/raw/Mounds_raster_mask_opened_closed.npy"
            train_dataset = DEMTilesDataset(
                dem_path=dem_path,
                mask_path=mask_path,
                tile_size=args.tile_size,
                stride=args.stride,
                split='train'
            )
            val_dataset = DEMTilesDataset(
                dem_path=dem_path,
                mask_path=mask_path,
                tile_size=args.tile_size,
                stride=args.stride,
                split='val'
            )
            
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
    if not args.save_path:
        from datetime import datetime
        now = datetime.now()
        args.save_path = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_model.pth"
        
    # Calculate class weights
    class_weights = 1.0 / (np.bincount(train_dataset.mask.flatten()) + 1e-6)
    print("Class weights: ", class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_index)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train with validation
    train_loop(
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        args.num_epochs,
        args.save_path
    )