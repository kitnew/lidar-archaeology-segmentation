
import numpy as np
import hydra
import torch
import sys


from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import optimal_threshold, Metrics

def evaluate_model(cfg, model, dataloader, device):
    metrics = Metrics(0,0,0,0,0,0,0,0,0)

    h, w = dataloader.dataset.data.shape[1:]
    tile_size = cfg.dataset.val.tile_size if hasattr(cfg.dataset.val, 'tile_size') else 256 # Fallback if not in config
    # If the dataset is tiled via a manifest, we don't need a stride here, 
    # but we need the output map size.


    map = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    map_count = torch.zeros_like(map)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", file=sys.stdout):
            images = batch["data"].to(device)
            coords = batch["coords"]

            outputs = model(images)
            outputs = outputs["out"] if isinstance(outputs, dict) else outputs

            for out, coord in zip(outputs, coords):
                # out is (1, TH, TW), y and x are scalars
                y, x = coord[0].item(), coord[1].item()
                tile_size_h, tile_size_w = out.shape[-2:]
                map[:, y:y+tile_size_h, x:x+tile_size_w] += torch.sigmoid(out)
                map_count[:, y:y+tile_size_h, x:x+tile_size_w] += 1

    final_map = map / torch.clamp(map_count, min=1)
    final_map = final_map.squeeze().cpu().numpy()
    
    # Create a mask of pixels that were actually covered by tiles during evaluation
    # to avoid penalizing metrics for areas not present in the validation split.
    evaluated_mask = (map_count > 0).squeeze().cpu().numpy()
    final_valid = dataloader.dataset.valid & evaluated_mask

    print(f"Final map stats: min={final_map.min():.4f}, max={final_map.max():.4f}, mean={final_map.mean():.4f}")
    
    best_t, metrics = optimal_threshold(final_map, dataloader.dataset.mask, valid=final_valid)
    print(f"\n--- Evaluation Results ---")
    print(f"Optimal Threshold: {best_t:.4f}")
    print(metrics)
    print("--------------------------\n")

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg):
    print("Initiating evaluation pipeline")
    device = torch.device("cuda")

    print("Initiating model")
    model = instantiate(cfg.model)
    print("Loading weights")
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    print("Initiating dataset")
    dataset = instantiate(cfg.dataset.val)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    evaluate_model(cfg, model, dataloader, device)

if __name__ == "__main__":
    main()