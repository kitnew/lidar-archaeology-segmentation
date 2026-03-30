import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.channel_stats import compute_channel_stats

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_path,
        manifest_path,
        subset="train",
        pos_only=False,
        transforms=None,
        min_valid_fraction=0.0,
    ):
        # --- manifest ---
        if manifest_path.endswith(".parquet"):
            df = pd.read_parquet(manifest_path)
        else:
            df = pd.read_csv(manifest_path)

        df = df[df["subset"] == subset]

        if min_valid_fraction > 0:
            df = df[df["valid_fraction"] >= min_valid_fraction]

        if pos_only:
            df = df[df["has_positive"] == True]

        self.df = df.sort_values(by="tile_id").reset_index(drop=True)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]

        # --- загрузка файла ---
        data_file = np.load(data_path, allow_pickle=True)

        self.data = data_file["data"]      # (C, H, W)
        self.mask = data_file["mask"]      # (H, W)
        self.valid = data_file["valid"]    # (H, W)

        # --- проверка формы ---
        assert self.data.ndim == 3, "data must be (C, H, W)"

        self.C, self.H, self.W = self.data.shape

        self.transforms = transforms

        self.mean = 0
        self.std = 0

        stats = compute_channel_stats(data_path, manifest_path, subset=("train" if subset != "full" else None), min_valid_fraction=min_valid_fraction)

        self.mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        y = int(row["y"])
        x = int(row["x"])
        tile_size = int(row["tile_size"])

        # --- вырезка тайла ---
        data_tile = self.data[:, y:y+tile_size, x:x+tile_size]
        mask_tile = self.mask[y:y+tile_size, x:x+tile_size]
        valid_tile = self.valid[y:y+tile_size, x:x+tile_size]

        # --- нормализация ---
        data_tile = (data_tile - self.mean) / (self.std + 1e-8)

        # --- типы ---
        data_tile = data_tile.astype(np.float32)
        mask_tile = mask_tile.astype(np.float32)[None, ...]
        valid_tile = valid_tile.astype(np.float32)[None, ...]

        # --- transforms ---
        if self.transforms:
            data_tile, mask_tile = self.transforms(data_tile, mask_tile)

        return {
            "data": torch.from_numpy(data_tile),   # (C, H, W)
            "mask": torch.from_numpy(mask_tile),    # (1, H, W)
            "valid": torch.from_numpy(valid_tile),   # (1, H, W)
            "coords": torch.tensor([y, x], dtype=torch.long)
        }