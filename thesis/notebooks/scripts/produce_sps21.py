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

radius = 2
kernel_radius = 10

f_name = f"SPS21(znorm)_{kernel_radius}"

import numpy as np

data_path = f"/home/nc225mj/lidar-archaeology-segmentation/data/raw/DEM21_opt.npz"

data = np.load(data_path)
#dem = data["data"].astype(np.float32)
#mask = data["mask"].astype(np.float32)
#soft_mask = data["soft_mask"].astype(np.float32)
valid = data["validMask"].astype(np.uint8)

import rasterio
data_dir = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/"

files = ["slope21_opt.tif", f"svf21_{kernel_radius}.tif", f"positive_openness21_{kernel_radius}.tif"]
arrs = [rasterio.open(data_dir+f).read(1).astype(np.float32) for f in files]
assert len({a.shape for a in arrs}) == 1, "Shapes mismatch"
SPS = np.stack(arrs, axis=0)  # (3, H, W)


print(np.isnan(SPS).sum())
print((~valid.astype(np.bool)).sum())

valid_bool = valid.astype(np.bool)
nan_mask = np.isnan(SPS).any(axis=0)  # (H, W)

valid = valid_bool & (~nan_mask)

SPS = np.nan_to_num(SPS, nan=no_data_value)

def zscore(x, mask):
    m = x[mask].mean()
    s = x[mask].std()
    x = (x - m) / (s + 1e-8)
    x[~mask] = no_data_value
    return x

for c in range(3):
    SPS[c] = zscore(SPS[c], valid)

# assert SPS[0].shape == mask.shape, f"{SPS[0].shape},{mask.shape}"
# assert SPS[1].shape == mask.shape, f"{SPS[1].shape},{mask.shape}"
# assert SPS[2].shape == mask.shape, f"{SPS[2].shape},{mask.shape}"
# 
# assert SPS[0].shape == soft_mask.shape, f"{SPS[0].shape},{soft_mask.shape}"
# assert SPS[1].shape == soft_mask.shape, f"{SPS[1].shape},{soft_mask.shape}"
# assert SPS[2].shape == soft_mask.shape, f"{SPS[2].shape},{soft_mask.shape}"

assert SPS[0].shape == valid.shape, f"{SPS[0].shape},{valid.shape}"
assert SPS[1].shape == valid.shape, f"{SPS[1].shape},{valid.shape}"
assert SPS[2].shape == valid.shape, f"{SPS[2].shape},{valid.shape}"

assert np.isnan(SPS).sum() == 0, np.isnan(SPS).sum()


file_structure["data"] = SPS
file_structure["mask"] = None
file_structure["soft_mask"] = None
file_structure["valid"] = valid

file_structure["meta"]["height"] = SPS[0].shape[0]
file_structure["meta"]["width"] = SPS[0].shape[1]
file_structure["meta"]["nodata_fill_value"] = no_data_value

file_structure["meta"]["created_from"]["data_path"] = data_path
file_structure["meta"]["created_from"]["mask_path"] = data_path

np.savez_compressed(f"/home/nc225mj/lidar-archaeology-segmentation/data/processed/{f_name}.npz", **file_structure)