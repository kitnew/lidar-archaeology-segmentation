from torch.utils.data import Dataset
import numpy as np
import torch

class DEMTilesDataset(Dataset):
    def __init__(self, dem_path, mask_path, tile_size=64, stride=32):
        """
        Simplified dataset for DEM and mask tiling.
        
        Args:
            dem_path: Path to DEM data .npz file
            mask_path: Path to mask .npy file
            tile_size: Size of the square tiles to extract
            stride: Stride for sliding window
        """
        # Load data
        data = np.load(dem_path)
        self.dem = data["dem"].astype(np.float32)
        self.valid = data["valid"].astype(bool)
        self.mask = np.load(mask_path).astype(np.int64)
        self.tile_size = tile_size
        self.stride = stride

        # Generate all tile coordinates
        H, W = self.dem.shape
        self.coords = []
        
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                # Check if tile has any valid DEM data
                if self.valid[y:y+tile_size, x:x+tile_size].any():
                    self.coords.append((y, x))
        
        print(f"Dataset: {len(self.coords)} tiles")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        dem_tile = self.dem[y:y+self.tile_size, x:x+self.tile_size]
        mask_tile = self.mask[y:y+self.tile_size, x:x+self.tile_size]
        valid_tile = self.valid[y:y+self.tile_size, x:x+self.tile_size]
        
        dem_tile = np.stack((dem_tile, dem_tile, dem_tile), axis=0)
        
        return {
            'dem': torch.from_numpy(dem_tile).float(),
            'valid': torch.from_numpy(valid_tile).bool(),
            'mask': torch.from_numpy(mask_tile).long(),
            'coords': np.array([y, x])
        }