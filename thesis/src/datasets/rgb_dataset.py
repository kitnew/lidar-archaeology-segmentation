from torch.utils.data import Dataset, Subset
import numpy as np
import torch

from torchvision.transforms import v2

augmentations = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(90.0),  # type: ignore
    v2.ToDtype(torch.float32, scale=True)
])

class RGBTilesDataset(Dataset):
    def __init__(self, rgb_path, mask_path, tile_size=64, stride=32, pos_only:bool = False, transforms: bool = False, no_gt: bool = False):
        """
        Simplified dataset for DEM and mask tiling.
        
        Args:
            rgb_path: Path to RGB data .npz file
            mask_path: Path to mask .npy file
            tile_size: Size of the square tiles to extract
            stride: Stride for sliding window
        """
        # Load data
        data = np.load(rgb_path)
        try:
            self.rgb = data["rgb"].astype(np.float32)
            self.valid = data["valid"].astype(bool)
        except Exception as e:
            self.rgb = data["dataset"].astype(np.float32)
            # Assuming rgb is (C, H, W) based on rasterio.read() and your use in EVALUATION
            self.valid = np.ones((self.rgb.shape[1], self.rgb.shape[2]), dtype=bool)
            
        if no_gt:
            self.mask = np.zeros((self.rgb.shape[1], self.rgb.shape[2]), dtype=np.uint8)
        else:
            self.mask = np.load(mask_path).astype(np.uint8)

        print(f"Dataset Initialized: RGB shape={self.rgb.shape}, Mask shape={self.mask.shape}, Valid shape={self.valid.shape}")
        self.tile_size = tile_size
        self.stride = stride
        self.pos_only = pos_only
        self.transforms = augmentations if transforms else None

        # Generate all tile coordinates
        H, W = self.rgb[0].shape
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
        rgb_tile = self.rgb[:, y:y+self.tile_size, x:x+self.tile_size]
        mask_tile = self.mask[y:y+self.tile_size, x:x+self.tile_size]
        valid_tile = self.valid[y:y+self.tile_size, x:x+self.tile_size]
        
        # Convert to tensor and prepare for transforms (forced independent memory)
        rgb_tensor = torch.from_numpy(np.array(rgb_tile, copy=True)).float()
        mask_tensor = torch.from_numpy(np.array(mask_tile, copy=True)).float().unsqueeze(0)
        valid_tensor = torch.from_numpy(np.array(valid_tile, copy=True)).bool().unsqueeze(0)

        # Apply the same transform to both image and mask
        if self.transforms is not None:
            # Stack all tensors to apply the same transform
            stacked = torch.cat([rgb_tensor, mask_tensor, valid_tensor.float()], dim=0)
            transformed = self.transforms(stacked)

            # Split back
            rgb_tensor = transformed[:3]  # First 3 channels are the image
            mask_tensor = transformed[3:4]  # Next channel is mask
            valid_tensor = transformed[4:].bool()  # Last channel is valid mask

        return {
            'data': rgb_tensor.clone(),
            'valid': valid_tensor.squeeze(0).clone(),  # Remove channel dim
            'mask': mask_tensor.squeeze(0).clone(),    # Remove channel dim
            'coords': torch.tensor([y, x], dtype=torch.int32)
        }