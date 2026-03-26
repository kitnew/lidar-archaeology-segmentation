import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys

from src.models.DeepLab_V3 import create_model
from src.datasets.dem_dataset import DEMTilesDataset
from src.datasets.rgb_dataset import RGBTilesDataset

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

def evaluate_model(model, dataloader, device, output_dir, dataset, num_samples=10, threshold=0.8, pred_map_size=None, no_gt=False):
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
            images = batch['data'].to(device)
            
            coords = batch.get("coords", None)
            
            # Get model predictions
            outputs = model(images)['out']

            for out, (y, x) in zip(outputs.squeeze(1), coords):
                map[:, y:y+tile_size, x:x+tile_size] += out.sigmoid()
                map_count[:, y:y+tile_size, x:x+tile_size] += 1

    final_map = map / torch.clamp(map_count, min=1)
    final_map = final_map.squeeze().cpu().numpy()

    pred_dir = os.path.join(output_dir, f"{dataset}_full_prediction_map.npy")
    os.makedirs(os.path.dirname(pred_dir), exist_ok=True)
    np.save(pred_dir, final_map)
    print(f"Full prediction map saved to: {pred_dir}")
    
    if not no_gt:
        metrics = calculate_metrics(final_map, global_mask, global_valid, threshold)
    else:
        metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'specificity': 0}
    return metrics

if __name__ == '__main__':
    import argparse
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser(description='Evaluate DeepLabV3 model on test set.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--tile-norm', action='store_true', help='Normalize tiles')
    parser.add_argument('--norm-constant', type=float, default=50.0, help='Normalization constant for tile normalization')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile size')
    parser.add_argument('--stride', type=int, default=1024, help='Stride for test set (use non-overlapping tiles)')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for binary prediction')
    parser.add_argument('--mode', type=str, choices=['slurm', 'local'], default='local', help='Execution mode')
    parser.add_argument('--dataset', type=str, choices=['DEM', 'DEM_hillshade', 'DEM_slope', 'DEM_hillshade_slope', 'DEM21_opt', 'DEM21_opt_hillshade', 'DEM21_opt_slope', 'DEM21_opt_hillshade_slope', 'MC', 'JZ', 'RGB', 'RGB21'], default='DEM', help='Dataset to evaluate on')
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

    dem_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{args.dataset.split('_')[0]}_normalized{f"_{args.split}" if args.split != 'full' else ""}.npz"
    #dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM.npz"
    mask_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/mounds_mask_shadowed{f"_{args.split}" if args.split != 'full' else ""}.npy"
    dem_hillshade_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{args.dataset.split('_')[0]}_hillshade_norm{f"_{args.split}" if args.split != 'full' else ""}.npy"
    dem_slope_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{args.dataset.split('_')[0]}_slope_norm{f"_{args.split}" if args.split != 'full' else ""}.npy"
    #rgb_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{args.dataset}{f"_{args.split}" if args.split != 'full' else "21"}.npz"
    rgb_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/RGB_normalized.npz"
    rgb21_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/RGB21_normalized.npz"

    if args.dataset != 'RGB' and args.dataset != 'RGB21':
        dataset = DEMTilesDataset(
            dem_path=dem_path,
            mask_path=mask_path,
            hillshade_path=dem_hillshade_path if args.dataset == 'DEM_hillshade' or args.dataset == 'DEM21_opt_hillshade' or args.dataset == 'DEM_hillshade_slope' or args.dataset == 'DEM21_opt_hillshade_slope' else None,
            slope_path=dem_slope_path if args.dataset == 'DEM_slope' or args.dataset == 'DEM21_opt_slope' or args.dataset == 'DEM_hillshade_slope' or args.dataset == 'DEM21_opt_hillshade_slope' else None,
            tile_size=args.tile_size,
            stride=args.stride,
            tile_norm=args.tile_norm,
            norm_constant=args.norm_constant,
            transforms=False,
            no_gt=args.no_gt
        )
    else:
        dataset = RGBTilesDataset(
            rgb_path=rgb_path if args.dataset == 'RGB' else rgb21_path,
            mask_path=mask_path,
            tile_size=args.tile_size,
            stride=args.stride,
            transforms=False,
            no_gt=args.no_gt
        )

    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    print(f'Test set: {len(dataset)} samples')
    print(f'Test set shape: {dataset[0]["data"].shape}')

    tile_size = args.tile_size
    stride = args.stride
    
    if args.dataset != 'RGB' and args.dataset != 'RGB21':
        dem_height = dataset.dem.shape[0]
        dem_width = dataset.dem.shape[1]
    else:
        dem_height = dataset.rgb.shape[1]
        dem_width = dataset.rgb.shape[2]
    
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
        dataset=args.dataset.split('_')[0],
        num_samples=args.num_samples,
        threshold=args.threshold,
        pred_map_size = (total_height, total_width),
        no_gt=args.no_gt
    )
    
    # Print results
    print('\nEvaluation Results:')
    print('-' * 40)
    for metric, value in metrics.items():
        print(f'{metric.capitalize()}: {value:.4f}')
    print('-' * 40)
    metrics_path = os.path.join(args.output_dir, f"{args.dataset.split('_')[0]}_metrics.json")
    
    # Save metrics to JSON file
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'Metrics saved to: {os.path.abspath(metrics_path)}')
    print(f'Predictions saved to: {os.path.abspath(args.output_dir)}')

