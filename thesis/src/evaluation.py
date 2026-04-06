
import numpy as np
import hydra
import torch
import sys


from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

from utils.metrics import optimal_threshold, Metrics

@dataclass
class EvaluationConfig:
    pass


def evaluate_model(cfg, model, dataloader, device):
    metrics = Metrics(0,0,0,0,0,0,0)

    h, w = dataloader.dataset.data.shape[1:]
    tile_size = cfg.dataset.full.tile_size if hasattr(cfg.dataset.full, 'tile_size') else 256 # Fallback if not in config
    # If the dataset is tiled via a manifest, we don't need a stride here, 
    # but we need the output map size.


    map = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    map_count = torch.zeros_like(map)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", file=sys.stdout):
            images = batch["data"].to(device)
            coords = batch["coords"]

            outputs = model(images)["out"]
            preds = torch.sigmoid(outputs).cpu().numpy()

            for out, (y, x) in zip(outputs, coords):
                # out is (1, TH, TW), y and x are scalars
                tile_size_h, tile_size_w = out.shape[-2:]
                map[:, y:y+tile_size_h, x:x+tile_size_w] += torch.sigmoid(out)
                map_count[:, y:y+tile_size_h, x:x+tile_size_w] += 1

    final_map = map / torch.clamp(map_count, min=1)
    final_map = final_map.squeeze().cpu().numpy()
    
    metrics = optimal_threshold(final_map, dataloader.dataset.mask)
    print(metrics)

@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg):
    print("Initiating evaluation pipeline")
    device = torch.device("cuda")

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    dataset = instantiate(cfg.dataset.full)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    evaluate_model(cfg, model, dataloader, device)

if __name__ == "__main__":
    main()