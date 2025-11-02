from torch.utils.data import Dataset, Subset
import numpy as np
import torch

from torchvision.transforms import v2

augmentations = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(90),
    v2.ToDtype(torch.float32, scale=True)
])

class DEMTilesDataset(Dataset):
    def __init__(self, dem_path, mask_path, tile_size=64, stride=32, pos_only:bool = False, transforms: bool = False, no_gt: bool = False):
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
        self.mask = np.load(mask_path).astype(np.uint8) if not no_gt else np.zeros((self.dem.shape[0], self.dem.shape[1]), dtype=np.uint8)
        self.tile_size = tile_size
        self.stride = stride
        self.pos_only = pos_only
        self.transforms = augmentations if transforms else None

        # Generate all tile coordinates
        H, W = self.dem.shape
        self.coords = []
        
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                if y + tile_size > H or x + tile_size > W:
                    continue
                # Check if tile has any valid DEM data
                if self.valid[y:y+tile_size, x:x+tile_size].any():
                    if self.pos_only:
                        if self.mask[y:y+tile_size, x:x+tile_size].any():
                            self.coords.append((y, x))
                    else:
                        self.coords.append((y, x))
                #else:
                #    print(f"Skipping tile at ({y}, {x}) - no valid DEM data")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        dem_tile = self.dem[y:y+self.tile_size, x:x+self.tile_size]
        mask_tile = self.mask[y:y+self.tile_size, x:x+self.tile_size]
        valid_tile = self.valid[y:y+self.tile_size, x:x+self.tile_size]
        
        dem_tile = np.stack((dem_tile, dem_tile, dem_tile), axis=0)
        
        # Convert to tensor and prepare for transforms
        dem_tensor = torch.from_numpy(dem_tile).float()
        mask_tensor = torch.from_numpy(mask_tile).float().unsqueeze(0)  # Add channel dim
        valid_tensor = torch.from_numpy(valid_tile).bool().unsqueeze(0)  # Add channel dim

        # Apply the same transform to both image and mask
        if self.transforms is not None:
            # Stack all tensors to apply the same transform
            stacked = torch.cat([dem_tensor, mask_tensor, valid_tensor.float()], dim=0)
            transformed = self.transforms(stacked)

            # Split back
            dem_tensor = transformed[:3]  # First 3 channels are the image
            mask_tensor = transformed[3:4]  # Next channel is mask
            valid_tensor = transformed[4:].bool()  # Last channel is valid mask

        return {
            'dem': dem_tensor,
            'valid': valid_tensor.squeeze(0),  # Remove channel dim
            'mask': mask_tensor.squeeze(0),    # Remove channel dim
            'coords': torch.tensor([y, x], dtype=torch.int32)
        }