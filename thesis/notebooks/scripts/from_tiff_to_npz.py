import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

rgb_path = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/ldr21_d_STU_General.tif"
dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM.tif"
dem21_path = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM21.tif"
dem_npz_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM.npz"

print("Loading DEM data...")
with rasterio.open(dem_path) as src_dem:
    dem_left, dem_bottom, dem_right, dem_top = src_dem.bounds

print("Loading DEM21 data...")
with rasterio.open(dem21_path) as src_dem:
    dem21 = src_dem.read(1).astype(np.float32)

print("Loading DEM npz data...")
dem_npz = np.load(dem_npz_path)
dem_valid = dem_npz["valid"]

print("Loading RGB data...")
with rasterio.open(rgb_path) as src_rgb:
    rgb = src_rgb.read().astype(np.float32)
    rgb_mask = src_rgb.dataset_mask().astype(bool)

    window = rasterio.windows.from_bounds(
        dem_left, dem_bottom, dem_right, dem_top,
        transform=src_rgb.transform
    )

    rgb_dem_crop = src_rgb.read(window=window).astype(np.float32)

rgb = rgb[:, :-1, :]
rgb_mask = rgb_mask[:-1, :]

print("Normalizing RGB data...")
rgb_norm = np.zeros_like(rgb, dtype=np.float32)
rgb_dem_norm = np.zeros_like(rgb_dem_crop, dtype=np.float32)

rgb_norm[:, rgb_mask] = rgb[:, rgb_mask] / 255.0
rgb_dem_norm[:, dem_valid] = rgb_dem_crop[:, dem_valid] / 255.0

rgb_norm[:, ~rgb_mask] = -1.0
rgb_dem_norm[:, ~dem_valid] = -1.0

print("Saving RGB normalized data...")
to_save = {
    "rgb": rgb_norm,
    "valid": rgb_mask.astype(np.uint8)
}

assert rgb_norm.shape == dem21.shape, f"Shape mismatch: rgb_norm {rgb_norm.shape} vs dem21 {dem21.shape}"

np.savez_compressed("/home/nc225mj/lidar-archaeology-segmentation/data/processed/RGB21_normalized.npz", **to_save)


to_save = {
    "rgb": rgb_dem_norm,
    "valid": dem_valid.astype(np.uint8)
}

np.savez_compressed("/home/nc225mj/lidar-archaeology-segmentation/data/processed/RGB_normalized.npz", **to_save)


h, w = rgb_dem_norm[0].shape

print(f"h: {h}, w: {w}")
print(f"RGB_DEM shape: {rgb_dem_norm.shape}")

train_start = int(0.20 * h)
val_end = train_start + int(0.20 * h)
#val_end = h
train_end = h

print(f"Train start: {train_start}")
print(f"Train end: {train_end}")
print(f"Val end: {val_end}")

# Split
print("Splitting data...")
rgb_dem_train_start = rgb_dem_norm[:, :train_start, :]
rgb_dem_train_end = rgb_dem_norm[:, val_end:train_end, :]
rgb_dem_train = np.concatenate((rgb_dem_train_start, rgb_dem_train_end), axis=1)
rgb_dem_val = rgb_dem_norm[:, train_start:val_end, :]

rgb_valid_train_start = dem_valid[:train_start, :]
rgb_valid_train_end = dem_valid[val_end:train_end, :]
rgb_valid_train = np.concatenate((rgb_valid_train_start, rgb_valid_train_end), axis=0)
rgb_valid_val = dem_valid[train_start:val_end, :]

rgb_dem_train = {"rgb": rgb_dem_train, "valid": rgb_valid_train}
rgb_dem_val   = {"rgb": rgb_dem_val, "valid": rgb_valid_val}
print("Saving split data...")
#np.savez_compressed("/home/nc225mj/lidar-archaeology-segmentation/data/processed/RGB0.80(4)_train.npz", **rgb_dem_train)
#np.savez_compressed("/home/nc225mj/lidar-archaeology-segmentation/data/processed/RGB0.80(4)_val.npz", **rgb_dem_val)