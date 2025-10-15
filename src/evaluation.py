import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch import nn

import sys

from src.models.DeepLab_V3 import create_model
from src.datasets.dem_dataset import DEMTilesDataset

IGNORE_INDEX = -1

def calculate_metrics(outputs, targets, valid=None, threshold=0.5):
    """Calculate metrics for confidence-based segmentation."""
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
        
        # Flatten tensors
        preds_flat = preds.view(-1).cpu().numpy()
        targets_flat = targets.view(-1).cpu().numpy()
        
        # Calculate accuracy
        accuracy = (preds_flat > threshold) == (targets_flat > 0.8)
        accuracy = accuracy.mean()
        
        # Calculate true/false positives/negatives
        tp = ((preds_flat > threshold) & (targets_flat > 0.8)).sum()
        fp = ((preds_flat > threshold) & (targets_flat <= 0.8)).sum()
        fn = ((preds_flat <= threshold) & (targets_flat > 0.8)).sum()
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Calculate MSE for confidence values
        mse = nn.MSELoss()(probs, targets).item()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mse': mse
        }

def visualize_sample(dem_tile, true_mask, pred_mask, output_dir, index, cmap='viridis'):
    """Visualize DEM and results with color coding:
    - Black: False Positive
    - White: True Positive
    - Black: True Negative
    - Red: False Negative
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot DEM
    axes[0].imshow(dem_tile, cmap='terrain')
    axes[0].set_title('DEM')
    axes[0].axis('off')
    
    # Create result visualization
    binary_pred = (pred_mask > 0.8).astype(float)
    result = np.zeros((*true_mask.shape, 3))  # RGB image
    
    # True Negative (black): 0 in both true and pred
    tn_mask = (true_mask == 0) & (binary_pred == 0)
    result[tn_mask] = [0, 0, 0]  # Black
    
    # False Positive (yellow): 0 in true, 1 in pred
    fp_mask = (true_mask == 0) & (binary_pred == 1)
    result[fp_mask] = [1, 1, 0]  # Yellow
    
    # False Negative (red): 1 in true, 0 in pred
    fn_mask = (true_mask == 1) & (binary_pred == 0)
    result[fn_mask] = [1, 0, 0]  # Red
    
    # True Positive (white): 1 in both true and pred
    tp_mask = (true_mask == 1) & (binary_pred == 1)
    result[tp_mask] = [1, 1, 1]  # White
    
    # Plot results
    axes[1].imshow(result)
    axes[1].set_title('Results (TP: white, FP: black, FN: red, TN: black)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_{index:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Plot DEM
    axes[0].imshow(dem_tile, cmap='terrain')
    axes[0].set_title('DEM')
    axes[0].axis('off')
    
    # Plot true mask
    axes[1].imshow(true_mask, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot predicted mask (binary)
    axes[2].imshow((pred_mask > 0.8).astype(float), cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title('Prediction (Binary)')
    axes[2].axis('off')
    
    # Plot predicted mask (confidence)
    conf = axes[3].imshow(pred_mask, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(conf, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title('Prediction (Confidence)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_old_{index:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def evaluate_model(model, dataloader, device, output_dir, num_samples=10):
    """Evaluate model on test set and generate visualizations."""
    model.eval()
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mse': []
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating', file=sys.stdout)):
            images = batch['dem'].to(device)
            masks = batch['mask'].to(device).unsqueeze(1)
            valid = batch['valid'].to(device).unsqueeze(1)
            
            # Get model predictions
            outputs = model(images)['out']
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks, valid)
            for key in metrics:
                if key in batch_metrics:
                    metrics[key].append(batch_metrics[key])
            
            # Save visualizations for first few samples
            if i < num_samples:
                for j in range(images.size(0)):
                    # Convert tensors to numpy arrays
                    dem = images[j].cpu().numpy().mean(0)  # Average across channels
                    true_mask = masks[j][0].cpu().numpy()  # Remove channel dim
                    pred_mask = torch.sigmoid(outputs[j][0]).cpu().numpy()  # Apply sigmoid and remove channel dim
                    
                    # Visualize
                    visualize_sample(
                        dem, true_mask, pred_mask,
                        output_dir=output_dir,
                        index=i * dataloader.batch_size + j
                    )
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    return avg_metrics

if __name__ == '__main__':
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
    model.load_state_dict(checkpoint)
    
    # Create test dataset and dataloader
    print('Loading test dataset...')
    if args.mode == 'slurm':
        dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM_normalized.npz"
        mask_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/mounds_mask_shadowed.npy"
    else:  # local
        dem_path = "/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/processed/DEM_normalized.npz"
        mask_path = "/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/processed/mounds_mask_shadowed.npy"
    
    dataset = DEMTilesDataset(
        dem_path=dem_path,
        mask_path=mask_path,
        tile_size=args.tile_size,
        stride=args.stride,
    )
    
    coords = np.array(dataset.coords)  # shape: (N, 2) -> [y, x]
    y_values = coords[:, 0]
    y_threshold = np.percentile(y_values, 70)

    train_indices = [i for i, (y, _) in enumerate(dataset.coords) if y < y_threshold]
    val_indices   = [i for i, (y, _) in enumerate(dataset.coords) if y >= y_threshold]
    test_indices  = [i for i, (y, _) in enumerate(dataset.coords) if y >= y_threshold]

    test_subset = Subset(dataset, test_indices)

    test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    print(f'Test set: {len(test_subset)} samples')
    
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
