import random
import numpy as np
import pandas as pd
import torch
import os
import sys

# Add the parent directory (src) to sys.path so we can import modules from it
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from datasets.dataset import SegmentationDataset

DATA_PATH = "../../data/processed/DEM_raw.npz"
MANIFEST_PATH = "../../data/processed/split4_manifest.parquet"

SUBSETS = ["train", "val"]
TILE_SIZE_EXPECTED = 128
MIN_VALID_FRACTION = 0.0
NUM_RANDOM_SAMPLES = 50


def load_manifest(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_source_npz(path: str):
    f = np.load(path, allow_pickle=True)
    data = f["data"]   # (C, H, W)
    mask = f["mask"]   # (H, W)
    valid = f["valid"] # (H, W)

    assert data.ndim == 3, f"Expected data shape (C,H,W), got {data.shape}"
    assert mask.ndim == 2, f"Expected mask shape (H,W), got {mask.shape}"
    assert valid.ndim == 2, f"Expected valid shape (H,W), got {valid.shape}"

    C, H, W = data.shape
    assert mask.shape == (H, W), f"Mask shape {mask.shape} != {(H, W)}"
    assert valid.shape == (H, W), f"Valid shape {valid.shape} != {(H, W)}"

    return data, mask, valid


def assert_manifest_basic(df: pd.DataFrame):
    required_cols = {
        "tile_id",
        "y",
        "x",
        "tile_size",
        "split_id",
        "subset",
        "valid_fraction",
        "positive_fraction",
        "has_positive",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Manifest missing columns: {missing}"

    assert len(df) > 0, "Manifest is empty"
    assert df["subset"].isin(["train", "val"]).all(), "Invalid subset values found"
    assert (df["tile_size"] > 0).all(), "tile_size must be > 0"
    assert (df["valid_fraction"] >= 0).all() and (df["valid_fraction"] <= 1).all(), "valid_fraction out of range"
    assert (df["positive_fraction"] >= 0).all() and (df["positive_fraction"] <= 1).all(), "positive_fraction out of range"
    assert df["has_positive"].isin([0, 1]).all(), "has_positive must be 0/1"


def assert_manifest_geometry(df: pd.DataFrame, H: int, W: int):
    for _, row in df.iterrows():
        y = int(row["y"])
        x = int(row["x"])
        tile_size = int(row["tile_size"])

        assert y >= 0 and x >= 0, f"Negative tile coords: {(y, x)}"
        assert y + tile_size <= H, f"Tile exceeds H: y={y}, tile_size={tile_size}, H={H}"
        assert x + tile_size <= W, f"Tile exceeds W: x={x}, tile_size={tile_size}, W={W}"


def assert_dataset_basic(dataset: SegmentationDataset, subset: str):
    assert len(dataset) > 0, f"{subset} dataset is empty"

    sample = dataset[0]
    assert isinstance(sample, dict), "Dataset item must be a dict"

    assert "data" in sample and "mask" in sample and "valid" in sample, "Missing keys in sample"

    image = sample["data"]
    mask = sample["mask"]
    valid = sample["valid"]

    assert isinstance(image, torch.Tensor), "image must be torch.Tensor"
    assert isinstance(mask, torch.Tensor), "mask must be torch.Tensor"
    assert isinstance(valid, torch.Tensor), "valid must be torch.Tensor"

    assert image.ndim == 3, f"image must be (C,H,W), got {tuple(image.shape)}"
    assert mask.ndim == 3, f"mask must be (1,H,W), got {tuple(mask.shape)}"
    assert valid.ndim == 3, f"valid must be (1,H,W), got {tuple(valid.shape)}"

    assert mask.shape[0] == 1, f"mask first dim must be 1, got {mask.shape[0]}"
    assert valid.shape[0] == 1, f"valid first dim must be 1, got {valid.shape[0]}"

    assert image.shape[1] == TILE_SIZE_EXPECTED and image.shape[2] == TILE_SIZE_EXPECTED, (
        f"Unexpected tile size in image: {tuple(image.shape)}"
    )
    assert mask.shape[1] == TILE_SIZE_EXPECTED and mask.shape[2] == TILE_SIZE_EXPECTED, (
        f"Unexpected tile size in mask: {tuple(mask.shape)}"
    )
    assert valid.shape[1] == TILE_SIZE_EXPECTED and valid.shape[2] == TILE_SIZE_EXPECTED, (
        f"Unexpected tile size in valid: {tuple(valid.shape)}"
    )

    assert image.dtype == torch.float32, f"image dtype must be float32, got {image.dtype}"
    assert mask.dtype == torch.float32, f"mask dtype must be float32, got {mask.dtype}"
    assert valid.dtype == torch.float32, f"valid dtype must be float32, got {valid.dtype}"


def assert_dataset_matches_source(
    dataset: SegmentationDataset,
    source_data: np.ndarray,
    source_mask: np.ndarray,
    source_valid: np.ndarray,
    num_samples: int = 50,
):
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(num_samples, n)]

    for idx in indices:
        sample = dataset[idx]
        row = dataset.df.iloc[idx]

        y = int(row["y"])
        x = int(row["x"])
        tile_size = int(row["tile_size"])

        expected_image = source_data[:, y:y+tile_size, x:x+tile_size].astype(np.float32)
        # Apply normalization matching the dataset's __getitem__
        expected_image = (expected_image - dataset.mean) / (dataset.std + 1e-8)
        
        expected_mask = source_mask[y:y+tile_size, x:x+tile_size].astype(np.float32)[None, ...]
        expected_valid = source_valid[y:y+tile_size, x:x+tile_size].astype(np.float32)[None, ...]

        image = sample["data"].cpu().numpy()
        mask = sample["mask"].cpu().numpy()
        valid = sample["valid"].cpu().numpy()

        assert image.shape == expected_image.shape, f"Image shape mismatch at idx={idx}"
        assert mask.shape == expected_mask.shape, f"Mask shape mismatch at idx={idx}"
        assert valid.shape == expected_valid.shape, f"Valid shape mismatch at idx={idx}"

        if not np.array_equal(image, expected_image):
            raise AssertionError(f"Image content mismatch at idx={idx}, y={y}, x={x}")
        if not np.array_equal(mask, expected_mask):
            raise AssertionError(f"Mask content mismatch at idx={idx}, y={y}, x={x}")
        if not np.array_equal(valid, expected_valid):
            raise AssertionError(f"Valid content mismatch at idx={idx}, y={y}, x={x}")


def assert_manifest_stats_match_source(
    df: pd.DataFrame,
    source_mask: np.ndarray,
    source_valid: np.ndarray,
    num_samples: int = 100,
    atol: float = 1e-8,
):
    n = len(df)
    indices = list(range(n))
    random.shuffle(indices)
    indices = indices[: min(num_samples, n)]

    for idx in indices:
        row = df.iloc[idx]

        y = int(row["y"])
        x = int(row["x"])
        tile_size = int(row["tile_size"])

        mask_tile = source_mask[y:y+tile_size, x:x+tile_size]
        valid_tile = source_valid[y:y+tile_size, x:x+tile_size]

        total_pixels = mask_tile.size
        valid_pixels = int(valid_tile.sum())

        if valid_pixels == 0:
            continue

        positive_pixels = int((mask_tile * valid_tile).sum())
        valid_fraction = valid_pixels / total_pixels
        positive_fraction = positive_pixels / valid_pixels
        has_positive = int(positive_pixels > 0)

        print("---- DEBUG TILE ----")
        print("idx:", idx)
        print("tile_id:", row["tile_id"])
        print("y, x:", y, x)
        print("tile_size:", tile_size)
        print("manifest valid_fraction:", row["valid_fraction"])
        print("recomputed valid_fraction:", valid_fraction)
        print("manifest positive_fraction:", row["positive_fraction"])
        print("recomputed positive_fraction:", positive_fraction)
        print("valid unique:", np.unique(valid_tile))
        print("mask unique:", np.unique(mask_tile))
        print("valid sum:", valid_tile.sum())
        print("mask*valid sum:", (mask_tile * valid_tile).sum())

        assert np.isclose(row["valid_fraction"], valid_fraction, atol=atol), (
            f"valid_fraction mismatch at idx={idx}: "
            f"{row['valid_fraction']} vs {valid_fraction}"
        )
        assert np.isclose(row["positive_fraction"], positive_fraction, atol=atol), (
            f"positive_fraction mismatch at idx={idx}: "
            f"{row['positive_fraction']} vs {positive_fraction}"
        )
        assert int(row["has_positive"]) == has_positive, (
            f"has_positive mismatch at idx={idx}: "
            f"{row['has_positive']} vs {has_positive}"
        )


def reconstruct_mask_from_dataset(dataset: SegmentationDataset, H: int, W: int):
    recon_mask = np.zeros((H, W), dtype=np.uint8)
    coverage = np.zeros((H, W), dtype=np.uint16)

    for idx in range(len(dataset)):
        row = dataset.df.iloc[idx]
        y = int(row["y"])
        x = int(row["x"])
        tile_size = int(row["tile_size"])

        mask_tile = dataset[idx]["mask"].cpu().numpy()[0].astype(np.uint8)

        # при overlap берём union по маске
        recon_mask[y:y+tile_size, x:x+tile_size] = np.maximum(
            recon_mask[y:y+tile_size, x:x+tile_size],
            mask_tile
        )
        coverage[y:y+tile_size, x:x+tile_size] += 1

    return recon_mask, coverage


def assert_reconstruction_matches_source(dataset: SegmentationDataset, source_mask: np.ndarray):
    _, H, W = dataset.data.shape
    recon_mask, coverage = reconstruct_mask_from_dataset(dataset, H, W)

    covered = coverage > 0
    assert covered.any(), "No covered pixels in reconstruction"

    if not np.array_equal(recon_mask[covered], source_mask[covered].astype(np.uint8)):
        mismatches = np.sum(recon_mask[covered] != source_mask[covered].astype(np.uint8))
        raise AssertionError(f"Reconstructed mask differs from source on {mismatches} covered pixels")

    print(f"[OK] Reconstruction matches source on covered pixels. Coverage ratio: {covered.mean():.6f}")


def run_subset_tests(subset: str, source_data: np.ndarray, source_mask: np.ndarray, source_valid: np.ndarray):
    print(f"\n===== TESTING SUBSET: {subset} =====")

    dataset = SegmentationDataset(
        data_path=DATA_PATH,
        manifest_path=MANIFEST_PATH,
        subset=subset,
        transforms=None,
        min_valid_fraction=MIN_VALID_FRACTION,
    )

    print(f"{subset} dataset length: {len(dataset)}")

    assert_dataset_basic(dataset, subset)
    print("[OK] Dataset basic structure")

    assert_dataset_matches_source(dataset, source_data, source_mask, source_valid, num_samples=NUM_RANDOM_SAMPLES)
    print("[OK] Random tile content matches source")

    assert_reconstruction_matches_source(dataset, source_mask)
    print("[OK] Binary reconstruction check")

    return dataset


def main():
    manifest = load_manifest(MANIFEST_PATH)
    source_data, source_mask, source_valid = load_source_npz(DATA_PATH)
    _, H, W = source_data.shape

    print("===== TESTING MANIFEST =====")
    assert_manifest_basic(manifest)
    print("[OK] Manifest basic columns and value ranges")

    assert_manifest_geometry(manifest, H, W)
    print("[OK] Manifest geometry")

    assert_manifest_stats_match_source(manifest, source_mask, source_valid, num_samples=100)
    print("[OK] Manifest per-tile statistics")

    train_dataset = run_subset_tests("train", source_data, source_mask, source_valid)
    val_dataset = run_subset_tests("val", source_data, source_mask, source_valid)

    print("\n===== SUMMARY =====")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print("All dataset tests passed.")


if __name__ == "__main__":
    main()