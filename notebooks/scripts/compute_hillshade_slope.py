dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/raw/DEM21_opt.npz"

import numpy as np

data = np.load(dem_path)
dem = data["dataset"].astype(np.float32)
valid = data["validMask"].astype(bool)

dem = np.where(valid == 1, dem, np.nan)

dem = np.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)

cellsize = 1.0

p = np.pad(dem, 1, mode="edge")

# ========= 3. Extract 3×3 windows ===========
z1 = p[:-2, :-2]
z2 = p[:-2, 1:-1]
z3 = p[:-2, 2:]
z4 = p[1:-1, :-2]
z5 = p[1:-1, 1:-1]
z6 = p[1:-1, 2:]
z7 = p[2:, :-2]
z8 = p[2:, 1:-1]
z9 = p[2:, 2:]

# ========= 4. Compute gradients ===========
dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8.0 * cellsize)
dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8.0 * cellsize)

# ========= 5. SLOPE ===========
slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
slope_deg = slope_rad * 180.0 / np.pi
slope_deg = np.nan_to_num(slope_deg, nan=0.0)

# ========= 6. ASPECT ===========
aspect = np.arctan2(dz_dy, -dz_dx)
aspect = aspect % (2 * np.pi)

# ========= 7. HILLSHADE ===========
azimuth_deg = 315.0
altitude_deg = 45.0

alt = altitude_deg * np.pi / 180.0
az  = ((360.0 - azimuth_deg) + 90.0) % 360.0 * np.pi / 180.0

hillshade = 255 * (
    np.cos(alt) * np.cos(slope_rad) +
    np.sin(alt) * np.sin(slope_rad) * np.cos(az - aspect)
)

hillshade = np.clip(hillshade, 0, 255).astype(np.float32)
hillshade = np.nan_to_num(hillshade, nan=0.0)

slope_norm = slope_deg / 90.0
hillshade_norm = hillshade / 255.0

# ========= 9. Save ===========
np.save("/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM21_slope_deg.npy", slope_deg)
np.save("/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM21_hillshade.npy", hillshade)
np.save("/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM21_hillshade_norm.npy", hillshade_norm)
np.save("/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM21_slope_norm.npy", slope_norm)