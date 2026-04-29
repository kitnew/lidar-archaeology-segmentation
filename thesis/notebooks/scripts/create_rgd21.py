import numpy as np

file_structure = {
    "data": np.float32,   # [H, W]
    "mask": np.uint8,     # [H, W]
    "soft_mask": np.float32, # [H, W]
    "valid": np.uint8,    # [H, W]
    "meta": {
        "height": int,
        "width": int,
        "nodata_fill_value": float,
        "created_from": {
            "data_path": str,
            "mask_path": str
        }
    }
}

no_data_value = 0.0

import rasterio
import numpy as np

rgb_path = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/ldr21_d_STU_General.tif"
dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/raw/DEM21_opt.npz"

data = np.load(dem_path)
dem = data["dataset"].astype(np.float32)
#mask = data["mask"].astype(np.float32)
#soft_mask = data["soft_mask"].astype(np.float32)
valid = data["validMask"].astype(np.uint8)

# 2. Load RGB and compute window
with rasterio.open(rgb_path) as src_rgb:
    rgb_crop = src_rgb.read()

# rgb_crop shape = (3, H, W)
print("RGB shape:", rgb_crop.shape)
print("DEM shape:", dem.shape)

H = min(rgb_crop.shape[1], dem.shape[0])
W = min(rgb_crop.shape[2], dem.shape[1])

rgb_crop = rgb_crop[:, :H, :W]
dem = dem[:H, :W]
valid = valid[:H, :W]

print("RGB shape:", rgb_crop.shape)
print("DEM shape:", dem.shape)
print("VALID shape:", valid.shape)

f_name = f"RGDem21(STU_norm)"

rgb_crop = rgb_crop.astype(np.float32)

assert rgb_crop[0].shape == dem.shape
assert rgb_crop.dtype == dem.dtype

valid_bool = valid.astype(bool)

rgb_crop = rgb_crop / 255.0

mean_rgb = np.array([
    rgb_crop[0][valid_bool].mean(),
    rgb_crop[1][valid_bool].mean(),
    rgb_crop[2][valid_bool].mean(),
], dtype=np.float32)

std_rgb = np.array([
    rgb_crop[0][valid_bool].std(),
    rgb_crop[1][valid_bool].std(),
    rgb_crop[2][valid_bool].std(),
], dtype=np.float32)

std_rgb = np.maximum(std_rgb, 1e-6)

print("mean_rgb:", mean_rgb)
print("std_rgb:", std_rgb)

mean_dem = dem[valid_bool].mean()
std_dem = dem[valid_bool].std()
std_dem = max(std_dem, 1e-6)

print("mean_dem:", mean_dem)
print("std_dem:", std_dem)

rgb_z = np.empty_like(rgb_crop, dtype=np.float32)

for c in range(3):
    rgb_z[c] = (rgb_crop[c] - mean_rgb[c]) / std_rgb[c]

dem_z = np.zeros_like(dem, dtype=np.float32)
dem_z[valid_bool] = (dem[valid_bool] - mean_dem) / std_dem
dem_z[~valid_bool] = no_data_value

assert dem_z.shape == rgb_z[0].shape

rgb_z[2] = dem_z
print(rgb_z.shape)

file_structure["data"] = rgb_z
file_structure["mask"] = None
file_structure["soft_mask"] = None
file_structure["valid"] = valid

file_structure["meta"]["height"] = rgb_z.shape[1]
file_structure["meta"]["width"] = rgb_z.shape[2]
file_structure["meta"]["nodata_fill_value"] = no_data_value

file_structure["meta"]["created_from"]["data_path"] = dem_path
file_structure["meta"]["created_from"]["mask_path"] = None

np.savez_compressed(f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{f_name}.npz", **file_structure)