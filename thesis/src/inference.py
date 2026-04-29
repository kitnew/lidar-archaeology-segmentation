import numpy as np
import hydra
import torch
import sys
import os
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_inference(cfg, model, dataloader, device):
    """
    Performs tiled inference and reconstructs the global probability map 
    by averaging overlapping predictions.
    """
    # Get global dimensions from the underlying dataset
    h, w = dataloader.dataset.data.shape[1:]
    
    # Initialize global map and count for averaging overlaps (on GPU if possible)
    pred_map = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    map_count = torch.zeros((1, h, w), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", file=sys.stdout):
            images = batch["data"].to(device)
            coords = batch["coords"] # (B, 2) -> [y, x]

            # Model output handling (DeepLabV3 returns a dict with "out")
            outputs = model(images)
            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            probs = torch.sigmoid(logits)

            # Accumulate predictions into the global map
            for prob, coord in zip(probs, coords):
                # prob is (1, TH, TW)
                y, x = coord[0].item(), coord[1].item()
                tile_h, tile_w = prob.shape[-2:]
                pred_map[:, y:y+tile_h, x:x+tile_w] += prob
                map_count[:, y:y+tile_h, x:x+tile_w] += 1

    # Average overlapping areas to get the final probability map
    final_map = pred_map / torch.clamp(map_count, min=1)
    
    # Return as single-channel NumPy array (H, W)
    return final_map.squeeze(0).cpu().numpy()

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg):
    print(f"Initiating inference pipeline")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate and load model
    print("Loading model architecture...")
    model = instantiate(cfg.model)
    
    print("Loading weights...")
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 2. Instantiate dataset
    # Handles cases where 'dataset' might point to a group of subsets or a single dataset config
    print("Initializing dataset...")
    if 'test' in cfg.dataset:
        dataset = instantiate(cfg.dataset.test)
        print(f"Using 'test' subset for inference.")
    elif 'val' in cfg.dataset:
        dataset = instantiate(cfg.dataset.val)
        print(f"Using 'val' subset for inference.")
    elif 'full' in cfg.dataset:
        dataset = instantiate(cfg.dataset.full)
        print(f"Using 'full' subset for inference.")
    else:
        dataset = instantiate(cfg.dataset)

    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False
    )

    print(f"Total tiles in dataloader: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: No tiles to process. Check your manifest and subset settings.")
        return

    # 3. Running inference
    prob_map = run_inference(cfg, model, dataloader, device)

    # 4. Save results
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    print(f"Saving global probability map to: {cfg.output_path}")
    np.save(cfg.output_path, prob_map)
    
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()