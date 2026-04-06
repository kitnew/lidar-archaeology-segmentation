
import numpy as np
import hydra
import torch
import sys

from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm import tqdm

def test_model(cfg, model, dataloader, device):

    h,w = dataloader.dataset.data.shape[1:]
    tile_size = 128
    stride = 32

    map = torch.zeros((1, h, w), dtype=torch.float32, device=device)
    map_count = torch.zeros_like(map)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", file=sys.stdout):
            images = batch["data"].to(device)
            coords = batch["coords"]

            outputs = model(images)["out"]
            preds = torch.sigmoid(outputs).cpu().numpy()

            for out, (y, x) in zip(outputs.squeeze(1), coords):
                map[:, y:y+tile_size, x:x+tile_size] += torch.sigmoid(out)
                map_count[:, y:y+tile_size, x:x+tile_size] += 1

    final_map = map / torch.clamp(map_count, min=1)
    final_map = final_map.squeeze().cpu().numpy()

    return final_map

@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg):
    print("Initiating inference pipeline")
    device = torch.device("cuda")

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()

    dataset = instantiate(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    print(f"Dataset loaded. Total tiles: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    map = test_model(cfg, model, dataloader, device)

    np.save(cfg.output_path, map)

if __name__ == "__main__":
    main()