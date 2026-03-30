import json
import numpy as np
import pandas as pd

def compute_channel_stats(
    data_path,
    manifest_path,
    subset: str | None = "train",
    min_valid_fraction: float = 0.0,
):
    if manifest_path.endswith(".parquet"):
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    if subset is not None:
        df = df[df["subset"] == subset]

    if min_valid_fraction > 0:
        df = df[df["valid_fraction"] >= min_valid_fraction]

    df = df.reset_index(drop=True)  # pyright: ignore[reportAttributeAccessIssue]

    data_file = np.load(data_path, allow_pickle=True)
    data = data_file["data"]    # (C, H, W)
    valid = data_file["valid"]  # (H, W)

    if data.ndim != 3:
        raise ValueError(f"Expected data shape (C, H, W), got {data.shape}")

    C, H, W = data.shape

    channel_sum = np.zeros(C, dtype=np.float64)
    channel_sum_sq = np.zeros(C, dtype=np.float64)
    channel_count = np.zeros(C, dtype=np.int64)

    channel_min = np.full(C, np.inf, dtype=np.float64)
    channel_max = np.full(C, -np.inf, dtype=np.float64)

    for _, row in df.iterrows():
        y = int(row["y"])  # pyright: ignore[reportArgumentType]
        x = int(row["x"])  # pyright: ignore[reportArgumentType]
        tile_size = int(row["tile_size"])  # pyright: ignore[reportArgumentType]

        data_tile = data[:, y:y+tile_size, x:x+tile_size]      # (C, h, w)
        valid_tile = valid[y:y+tile_size, x:x+tile_size] > 0   # (h, w)

        if not np.any(valid_tile):
            continue

        for c in range(C):
            values = data_tile[c][valid_tile]

            if values.size == 0:
                continue

            channel_sum[c] += values.sum()
            channel_sum_sq[c] += np.square(values).sum()
            channel_count[c] += values.size

            vmin = values.min()
            vmax = values.max()

            if vmin < channel_min[c]:
                channel_min[c] = vmin
            if vmax > channel_max[c]:
                channel_max[c] = vmax

    if np.any(channel_count == 0):
        raise ValueError("At least one channel has zero valid pixels in selected subset.")

    mean = channel_sum / channel_count
    var = (channel_sum_sq / channel_count) - np.square(mean)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    stats = {
        "method": "zscore",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": channel_min.tolist(),
        "max": channel_max.tolist(),
        "count": channel_count.tolist(),
    }

    return stats