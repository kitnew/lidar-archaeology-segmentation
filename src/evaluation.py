import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch import nn
from skimage.morphology import opening, disk

import sys

from src.models.DeepLab_V3 import create_model
from src.datasets.dem_dataset import DEMTilesDataset

IGNORE_INDEX = -1

def calculate_metrics(probs, targets, valid=None, threshold=0.8):
    """Calculate metrics for binary semantic segmentation.
    
    Args:
        probs: Model output logits (after sigmoid) of shape (B, 1, H, W)
        targets: Ground truth masks of shape (B, 1, H, W)
        valid: Optional mask indicating valid pixels
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary containing various segmentation metrics
    """

    def compute_metrics(pred_bin, gt):
        tp = np.logical_and(pred_bin == 1, gt == 1).sum()
        fp = np.logical_and(pred_bin == 1, gt == 0).sum()
        fn = np.logical_and(pred_bin == 0, gt == 1).sum()
        tn = np.logical_and(pred_bin == 0, gt == 0).sum()

        iou = tp / (tp + fp + fn + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        mor10r = tp / (tp + fn + 10*fp + 1e-8)
        return iou, acc, mor10r

    best_t, best_iou, best_acc, best_mor = 0, 0, 0, 0
    thresholds = np.linspace(0.5, 0.999, 100)

    for t in thresholds:
        pred_bin = (probs > t).astype(np.uint8)
        iou, acc, mor = compute_metrics(pred_bin, targets)
        if iou > best_iou:
            best_t, best_iou, best_acc, best_mor = t, iou, acc, mor

    print(f"Optimal threshold: {best_t:.2f}")
    print(f"IoU={best_iou:.3f}, Accuracy={best_acc:.3f}, MOR10R={best_mor:.3f}")

    # Create binary predictions
    preds = (probs > best_t).astype(np.uint8)
    
    tp = np.logical_and(preds == 1, targets == 1).sum()
    fp = np.logical_and(preds == 1, targets == 0).sum()
    fn = np.logical_and(preds == 0, targets == 1).sum()
    tn = np.logical_and(preds == 0, targets == 0).sum()
    
    # Pixel-level metrics
    eps = 1e-10
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    
    # Class-wise metrics
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    # IoU (Jaccard Index) for each class
    iou_positive = tp / (tp + fp + fn + eps)
    
    return {
        # Overall metrics
        'accuracy': accuracy,   
        
        # Class-specific metrics
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou_positive,
        
        # Additional metrics
        'specificity': tn / (tn + fp + eps),  # True Negative Rate
    }

def evaluate_model(model, dataloader, device, output_dir, num_samples=10, threshold=0.8, pred_map_size=None):
    """Evaluate model on test set and generate visualizations."""
    model.eval()
    metrics ={
        'accuracy': 0,   
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'iou': 0,
        'specificity': 0
    }

    global_mask = dataloader.dataset.mask.astype(np.uint8)
    global_valid = dataloader.dataset.valid

    map = torch.zeros((1, total_height, total_width), dtype=torch.float32, device=device)
    map_count = torch.zeros_like(map)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating', file=sys.stdout)):
            images = batch['dem'].to(device)
            coords = batch.get("coords", None)
            
            # Get model predictions
            outputs = model(images)['out']

            for out, (y, x) in zip(outputs.squeeze(1), coords):
                map[:, y:y+tile_size, x:x+tile_size] += out.sigmoid()
                map_count[:, y:y+tile_size, x:x+tile_size] += 1

    final_map = map / torch.clamp(map_count, min=1)
    final_map = final_map.squeeze().cpu().numpy()

    pred_dir = os.path.join(output_dir, "full_prediction_map.npy")
    os.makedirs(os.path.dirname(pred_dir), exist_ok=True)
    np.save(pred_dir, final_map)
    print(f"Full prediction map saved to: {pred_dir}")
    
    metrics = calculate_metrics(final_map, global_mask, global_valid, threshold)
    return metrics

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
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for binary prediction')
    parser.add_argument('--mode', type=str, choices=['slurm', 'local'], default='local', help='Execution mode')
    parser.add_argument('--dataset', type=str, choices=['DEM', 'DEM21_opt'], default='DEM', help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'full'], default='test', help='Split to evaluate on')
    parser.add_argument('--no-gt', action='store_true', help='Do not use ground truth for evaluation')

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

    dem_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{args.dataset}_normalized{f"_{args.split}" if args.split != 'full' else ""}.npz"
    mask_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/mounds_mask_shadowed{f"_{args.split}" if args.split != 'full' else ""}.npy"
    
    dataset = DEMTilesDataset(
        dem_path=dem_path,
        mask_path=mask_path,
        tile_size=args.tile_size,
        stride=args.stride,
        transforms=False,
        no_gt=args.no_gt
    )

    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    print(f'Test set: {len(dataset)} samples')

    tile_size = args.tile_size
    stride = args.stride
    
    dem_height = dataset.dem.shape[0]
    dem_width = dataset.dem.shape[1]
    
    total_height = dem_height
    total_width  = dem_width
    print(f"Total height: {total_height}, total width: {total_width}")

    # Run evaluation
    print('Starting evaluation...')
    metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
        pred_map_size = (total_height, total_width)
    )
    
    # Print results
    print('\nEvaluation Results:')
    print('-' * 40)
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}: {value:.4f}')
    print('-' * 40)
    print(f'Visualizations saved to: {os.path.abspath(args.output_dir)}')
