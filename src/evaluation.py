import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.DeepLab_V3 import create_model
from src.datasets.dem_dataset import DEMTilesDataset

def calculate_metrics(outputs, targets, ignore_index=255):
    """Calculate pixel accuracy, precision, recall, and F1 score."""
    with torch.no_grad():
        _, preds = torch.max(outputs, 1)
        valid_pixels = (targets != ignore_index)
        
        # Flatten tensors
        preds_flat = preds[valid_pixels].cpu().numpy()
        targets_flat = targets[valid_pixels].cpu().numpy()
        
        # Calculate metrics
        accuracy = (preds_flat == targets_flat).mean()
        
        # Calculate true positives, false positives, false negatives
        tp = ((preds_flat == 1) & (targets_flat == 1)).sum()
        fp = ((preds_flat == 1) & (targets_flat == 0)).sum()
        fn = ((preds_flat == 0) & (targets_flat == 1)).sum()
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def visualize_sample(dem_tile, true_mask, pred_mask, output_dir, index, cmap='viridis'):
    """Visualize DEM, true mask, and predicted mask side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot DEM
    axes[0].imshow(dem_tile, cmap='terrain')
    axes[0].set_title('DEM')
    axes[0].axis('off')
    
    # Plot true mask
    axes[1].imshow(true_mask, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{index:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model(model, dataloader, device, output_dir, num_samples=10):
    """Evaluate model on test set and generate visualizations."""
    model.eval()
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Get model predictions
            outputs = model(images)['out']
            _, preds = torch.max(outputs, 1)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks)
            for key in metrics:
                metrics[key].append(batch_metrics[key])
            
            # Save visualizations for first few samples
            if i < num_samples:
                for j in range(images.size(0)):
                    # Convert tensors to numpy arrays
                    dem = images[j].cpu().numpy().mean(0)  # Average across channels for visualization
                    true_mask = (masks[j].cpu().numpy() > 0.5).astype(float)
                    pred_mask = (preds[j].cpu().numpy() > 0.5).astype(float)
                    
                    # Visualize
                    visualize_sample(
                        dem, true_mask, pred_mask,
                        output_dir=output_dir,
                        index=i * dataloader.batch_size + j
                    )
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    return avg_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DeepLabV3 model on test set.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile size')
    parser.add_argument('--stride', type=int, default=1024, help='Stride for test set (use non-overlapping tiles)')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--mode', type=str, choices=['slurm', 'local'], default='local', help='Execution mode')
    
    args = parser.parse_args()
    
    match(args.mode):
        case 'slurm':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        case 'local':
            device = torch.device('cpu')
        case _:
            raise ValueError(f"Unknown device: {args.device}")
    print("Device: ", device)
    
    # Load model
    print('Loading model...')
    model = create_model(backbone='resnet101', pretrained=False, eval=True).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset and dataloader
    print('Loading test dataset...')
    if args.mode == 'slurm':
        dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/raw/DEM.npz"
        mask_path = "/home/nc225mj/lidar-archaeology-segmentation/data/raw/Mounds_raster_mask_opened_closed.npy"
    else:  # local
        dem_path = "/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/raw/DEM.npz"
        mask_path = "/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/raw/Mounds_raster_mask_opened_closed.npy"
    
    test_dataset = DEMTilesDataset(
        dem_path=dem_path,
        mask_path=mask_path,
        tile_size=args.tile_size,
        stride=args.stride,
        split='test'
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    print(f'Test set: {len(test_dataset)} samples')
    
    # Run evaluation
    print('Starting evaluation...')
    metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Print results
    print('\nEvaluation Results:')
    print('-' * 40)
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}: {value:.4f}')
    print('-' * 40)
    print(f'Visualizations saved to: {os.path.abspath(args.output_dir)}')

if __name__ == '__main__':
    main()