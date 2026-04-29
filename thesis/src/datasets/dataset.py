import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.channel_stats import compute_channel_stats
from torchvision.transforms import v2
import random
import os
from collections.abc import Mapping


class RandomRotation(torch.nn.Module):
    def __init__(self, degrees: list[int] = [0, 90, 180, 270]):
        super().__init__()
        self.degrees = degrees

    def forward(self, data_tile, mask_tile, valid_tile):
        k = random.choice(self.degrees) // 90
        data_tile = torch.rot90(data_tile, k=k, dims=(-2, -1))
        mask_tile = torch.rot90(mask_tile, k=k, dims=(-2, -1))
        valid_tile = torch.rot90(valid_tile, k=k, dims=(-2, -1))
        return data_tile, mask_tile, valid_tile

augmentations = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    RandomRotation([0, 90, 180, 270]),
    #v2.GaussianNoise(sigma=5e-4),
    v2.ToDtype(torch.float32, scale=True)
])

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_path,
        manifest_path,
        subset="train",
        pos_only=False,
        transforms=False,
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
        self.valid = data_file["valid"]    # (H, W)
        
        ## use soft mask for training, hard mask for validation
        if (subset == "train" and "soft_mask" in data_file 
            and data_file["soft_mask"] is not None 
            and getattr(data_file["soft_mask"], "ndim", 0) >= 2):
            
            self.mask = data_file["soft_mask"].astype(np.float32)      # (H, W)
        else:
            self.mask = data_file.get("mask").astype(np.uint8) if subset != "full" else np.zeros_like(self.data[0], dtype=np.uint8)      # (H, W)

        # --- проверка формы ---
        assert self.data.ndim == 3, "data must be (C, H, W)"

        self.C, self.H, self.W = self.data.shape

        self.transforms = transforms if subset == "train" else None

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

        # --- типы ---
        data_tile = data_tile.astype(np.float32)
        mask_tile = mask_tile.astype(np.float32)[None, ...]
        valid_tile = valid_tile.astype(np.float32)[None, ...]

        # --- transforms ---
        if self.transforms:
            data_tile, mask_tile, valid_tile = self.transforms(data_tile, mask_tile, valid_tile)

        return {
            "data": torch.from_numpy(data_tile),   # (C, H, W)
            "mask": torch.from_numpy(mask_tile),    # (1, H, W)
            "valid": torch.from_numpy(valid_tile),   # (1, H, W)
            "coords": torch.tensor([y, x], dtype=torch.long)
        }

class TerassesSegmentationDataset(Dataset):
    def __init__(
        self,
        data_path,
        manifest_path,
        subset="train",
        pos_only=False,
        transforms=False,
        min_valid_fraction=0.0,
        held_out_location=None,
        full_location=None,
        use_soft_mask=False,
        neg_to_pos_ratio=None,
        random_state=42,
    ):
        """
        data_path:
            либо str -> путь к одному npz (для совместимости),
            либо dict[str, str] -> {location_id: npz_path} для master manifest

        manifest_path:
            master_manifest.parquet / .csv

        subset:
            "train" | "val" | "full"

        held_out_location:
            какая локация используется как val в текущем LOOCV fold
            обязательно для subset in {"train", "val"}.
            для subset="full" тоже желательно передавать, чтобы stats считались по train fold.

        full_location:
            локация для полного инференса, например "tribec"

        pos_only:
            оставить только positive tiles

        transforms:
            применяются только для train

        min_valid_fraction:
            фильтр по manifest

        use_soft_mask:
            если True, то для train использовать soft_mask при наличии

        neg_to_pos_ratio:
            только для train.
            Если задано, то negative тайлы будут подсемплированы
            до соотношения примерно neg_to_pos_ratio : 1 относительно positive.
            Например 2.0 => максимум 2 negative на 1 positive.

        random_state:
            seed для подсемплирования negative tiles
        """

        # -------------------------
        # LOAD MANIFEST
        # -------------------------
        if manifest_path.endswith(".parquet"):
            full_df = pd.read_parquet(manifest_path)
        else:
            full_df = pd.read_csv(manifest_path)

        if min_valid_fraction > 0:
            full_df = full_df[full_df["valid_fraction"] >= min_valid_fraction]

        if subset in ("train", "val") and held_out_location is None:
            raise ValueError("held_out_location must be provided for subset='train' or subset='val'")

        if subset == "full" and full_location is None:
            # если full_location не задан, берём все строки с use_for_loocv == 0
            df = full_df[full_df["use_for_loocv"] == 0].copy()
        elif subset == "full":
            df = full_df[full_df["location_id"] == full_location].copy()
        elif subset == "train":
            df = full_df[
                (full_df["use_for_loocv"] == 1) &
                (full_df["location_id"] != held_out_location)
            ].copy()
        elif subset == "val":
            df = full_df[
                (full_df["use_for_loocv"] == 1) &
                (full_df["location_id"] == held_out_location)
            ].copy()
        else:
            raise ValueError(f"Unknown subset: {subset}")

        if pos_only:
            df = df[df["has_positive"] == 1].copy()

        # -------------------------
        # OPTIONAL NEGATIVE SUBSAMPLING
        # -------------------------
        if subset == "train" and (not pos_only) and (neg_to_pos_ratio is not None):
            pos_df = df[df["has_positive"] == 1].copy()
            neg_df = df[df["has_positive"] == 0].copy()

            if len(pos_df) > 0 and len(neg_df) > 0:
                n_neg = min(len(neg_df), int(np.ceil(len(pos_df) * float(neg_to_pos_ratio))))
                neg_df = neg_df.sample(n=n_neg, random_state=random_state)
                df = pd.concat([pos_df, neg_df], axis=0)

        # сортировка
        if "global_tile_id" in df.columns:
            df = df.sort_values(by="global_tile_id").reset_index(drop=True)
        else:
            df = df.sort_values(by=["location_id", "tile_id"]).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("Filtered dataset is empty. Check subset/location/filters.")

        self.df = df
        self.subset = subset
        self.transforms = transforms if subset == "train" else None
        self.held_out_location = held_out_location
        self.full_location = full_location
        self.use_soft_mask = use_soft_mask

        # -------------------------
        # RESOLVE DATA PATHS
        # -------------------------
        if isinstance(data_path, (str, os.PathLike)):
            unique_locations = self.df["location_id"].unique().tolist()
            if len(unique_locations) != 1:
                raise ValueError(
                    "data_path is a single file, but manifest contains multiple locations. "
                    "Pass data_path as dict: {location_id: npz_path}"
                )
            self.data_paths = {unique_locations[0]: str(data_path)}
        elif isinstance(data_path, (dict, Mapping)):
            self.data_paths = dict(data_path)
        else:
            raise TypeError(f"data_path must be str or dict[location_id, path], got {type(data_path)}")


        # -------------------------
        # DETERMINE WHICH LOCATIONS MUST BE LOADED
        # -------------------------
        needed_locations = set(self.df["location_id"].unique().tolist())

        missing_locations = [loc for loc in needed_locations if loc not in self.data_paths]
        if len(missing_locations) > 0:
            raise ValueError(f"Missing npz paths for locations: {missing_locations}")

        # -------------------------
        # LOAD NPZ CACHE
        # -------------------------
        self.data_cache = {}
        self.mask_cache = {}
        self.valid_cache = {}
        self.meta_cache = {}

        for location_id in sorted(needed_locations):
            npz_path = self.data_paths[location_id]
            data_file = np.load(npz_path, allow_pickle=True)

            data_arr = data_file["data"].astype(np.float32)     # [C, H, W]
            valid_arr = data_file["valid"].astype(np.uint8)     # [H, W]

            if data_arr.ndim != 3:
                raise ValueError(f"{location_id}: data must be (C, H, W), got {data_arr.shape}")

            if subset == "train" and use_soft_mask and ("soft_mask" in data_file):
                soft = data_file["soft_mask"]
                if soft is not None and getattr(soft, "ndim", 0) >= 2:
                    mask_arr = soft.astype(np.float32)
                else:
                    mask_arr = data_file["mask"].astype(np.float32)
            elif subset == "full":
                mask_arr = np.zeros_like(valid_arr, dtype=np.float32)
            else:
                mask_arr = data_file["mask"].astype(np.float32)

            self.data_cache[location_id] = data_arr
            self.valid_cache[location_id] = valid_arr
            self.mask_cache[location_id] = mask_arr
            self.meta_cache[location_id] = data_file["meta"] if "meta" in data_file else None

        # -------------------------
        # SHAPE CHECKS FOR LOCATIONS USED BY CURRENT SUBSET
        # -------------------------
        for location_id in self.df["location_id"].unique():
            data_arr = self.data_cache[location_id]
            valid_arr = self.valid_cache[location_id]
            mask_arr = self.mask_cache[location_id]

            C, H, W = data_arr.shape

            if valid_arr.shape != (H, W):
                raise ValueError(
                    f"{location_id}: valid shape {valid_arr.shape} does not match data spatial shape {(H, W)}"
                )
            if mask_arr.shape != (H, W):
                raise ValueError(
                    f"{location_id}: mask shape {mask_arr.shape} does not match data spatial shape {(H, W)}"
                )

        # число каналов преедполагатся одинаковым внутри выбранного варианта
        current_locations = self.df["location_id"].unique().tolist()
        channel_counts = [self.data_cache[loc].shape[0] for loc in current_locations]
        if len(set(channel_counts)) != 1:
            raise ValueError(f"Channel count mismatch across locations: {dict(zip(current_locations, channel_counts))}")

        self.C = channel_counts[0]

    @property
    def data(self):
        loc = self.held_out_location if self.subset == "val" else self.full_location
        if loc is None:
            if len(self.data_cache) == 1:
                return next(iter(self.data_cache.values()))
            raise AttributeError(f"subset is '{self.subset}' and no location is specified, and multiple locations exist in cache.")
        return self.data_cache[loc]

    @property
    def mask(self):
        loc = self.held_out_location if self.subset == "val" else self.full_location
        if loc is None:
            if len(self.mask_cache) == 1:
                return next(iter(self.mask_cache.values()))
            raise AttributeError(f"subset is '{self.subset}' and no location is specified, and multiple locations exist in cache.")
        return self.mask_cache[loc]

    @property
    def valid(self):
        loc = self.held_out_location if self.subset == "val" else self.full_location
        if loc is None:
            if len(self.valid_cache) == 1:
                return next(iter(self.valid_cache.values()))
            raise AttributeError(f"subset is '{self.subset}' and no location is specified, and multiple locations exist in cache.")
        return self.valid_cache[loc]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        location_id = row["location_id"]
        y = int(row["y"])
        x = int(row["x"])
        tile_size = int(row["tile_size"])

        data_arr = self.data_cache[location_id]
        mask_arr = self.mask_cache[location_id]
        valid_arr = self.valid_cache[location_id]

        # --- tile crop ---
        data_tile = data_arr[:, y:y+tile_size, x:x+tile_size]
        mask_tile = mask_arr[y:y+tile_size, x:x+tile_size]
        valid_tile = valid_arr[y:y+tile_size, x:x+tile_size]

        # --- dtypes / shapes ---
        data_tile = data_tile.astype(np.float32)
        mask_tile = mask_tile.astype(np.float32)[None, ...]
        valid_tile = valid_tile.astype(np.float32)[None, ...]

        # --- transforms only for train ---
        if self.transforms:
            data_tile, mask_tile, valid_tile = self.transforms(data_tile, mask_tile, valid_tile)

        return {
            "data": torch.from_numpy(data_tile),    # (C, H, W)
            "mask": torch.from_numpy(mask_tile),    # (1, H, W)
            "valid": torch.from_numpy(valid_tile),  # (1, H, W)
            "coords": torch.tensor([y, x], dtype=torch.long),
        }