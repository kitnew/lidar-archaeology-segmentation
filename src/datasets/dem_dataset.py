from torch.utils.data import Dataset, random_split
import numpy as np
import torch

class DEMTilesDataset(Dataset):
    def __init__(self, dem_path, mask_path, tile_size=1024, stride=512, split='train', 
                 val_ratio=0.15, test_ratio=0.15, random_seed=782):
        """
        Args:
            dem_path: Path to DEM data .npz file
            mask_path: Path to mask .npy file
            tile_size: Size of the square tiles to extract
            stride: Stride for sliding window
            split: One of 'train', 'val', or 'test'
            val_ratio: Fraction of data to use for validation
            test_ratio: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
        """
        data = np.load(dem_path)
        self.dem = data["dataset"].astype(np.float32)
        self.valid = data["validMask"].astype(bool)
        self.mask = np.load(mask_path).astype(np.uint8)
        self.tile_size = tile_size
        self.stride = stride
        self.split = split

        # Generate all valid tile coordinates
        H, W = self.dem.shape
        coords = []
        for y in range(0, H - tile_size + 1, stride):
            for x in range(0, W - tile_size + 1, stride):
                if self.valid[y:y+tile_size, x:x+tile_size].sum() > 0:
                    coords.append((y, x))
        
        # Store all coordinates
        self.all_coords = coords
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        
        # Calculate split sizes
        n_total = len(coords)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val - n_test
        
        # Split indices
        indices = torch.randperm(n_total).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Select the appropriate split
        if split == 'train':
            self.coords = [coords[i] for i in train_indices]
        elif split == 'val':
            self.coords = [coords[i] for i in val_indices]
        elif split == 'test':
            self.coords = [coords[i] for i in test_indices]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
            
        print(f"{split.capitalize()} dataset: {len(self.coords)} samples")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        dem_tile = self.dem[y:y+self.tile_size, x:x+self.tile_size].copy()  # Копируем, чтобы не менять оригинал
        mask_tile = self.mask[y:y+self.tile_size, x:x+self.tile_size]
        valid_tile = self.valid[y:y+self.tile_size, x:x+self.tile_size]
        
        # Нормализация по валидным пикселям
        v = dem_tile[valid_tile]
        if v.size > 0:
            v_min, v_max = v.min(), v.max()
            scale = v_max - v_min
            if scale > 1e-6:
                dem_tile = (dem_tile - v_min) / (scale + 1e-6)
            else:
                dem_tile = np.zeros_like(dem_tile)
        else:
            dem_tile = np.zeros_like(dem_tile)

        # Заполняем невалидные пиксели нулями
        dem_tile[~valid_tile] = 0.0

        # Ограничиваем диапазон и приводим тип
        dem_tile = np.clip(dem_tile, 0.0, 1.0).astype(np.float32)

        # Репликация в 3 канала
        img = np.stack([dem_tile] * 3, axis=0)  # (3, H, W)

        # Маска: невалидные пиксели -> 255
        target = mask_tile.astype(np.int64)
        target[~valid_tile] = 255

        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(target)}