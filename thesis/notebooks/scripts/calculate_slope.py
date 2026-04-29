import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


# =========================
# INPUT
# =========================
#dem_path = "/home/nikitachernysh/Storage/Projects/lidar-archaeology-segmentation/data/georef/DEM.tif"
dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM21_opt.tif"

# Единицы slope: "degree" или "radian"
slope_units = "degree"

import rasterio

with rasterio.open(dem_path) as src:
    print("CRS:", src.crs)
    print("Is projected:", src.crs.is_projected)
    print("Linear units:", src.crs.linear_units)
    print("Transform:", src.transform)

# =========================
# HELPERS
# =========================
def read_dem(path: str):
    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        nodata = src.nodata

        # размер пикселя
        res_x = abs(transform.a)
        res_y = abs(transform.e)

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    return dem, profile, nodata, res_x, res_y


def save_raster(path: str, array: np.ndarray, profile: dict, nodata_value=np.nan):
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        compress="deflate",
        predictor=3
    )

    arr = array.astype(np.float32)

    if np.isnan(nodata_value):
        out_profile["nodata"] = np.nan
    else:
        out_profile["nodata"] = float(nodata_value)
        arr = np.where(np.isnan(arr), nodata_value, arr)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr, 1)


def compute_slope(dem: np.ndarray, res_x: float, res_y: float, units: str = "degree"):
    """
    Вычисление slope через центральные разности.
    Возвращает slope в градусах или радианах.
    """
    valid = np.isfinite(dem)
    filled = np.where(valid, dem, np.nan)

    dz_dy, dz_dx = np.gradient(filled, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    if units == "degree":
        slope = np.degrees(slope_rad)
    elif units == "radian":
        slope = slope_rad
    else:
        raise ValueError("units must be 'degree' or 'radian'")

    slope[~valid] = np.nan
    return slope.astype(np.float32)

# =========================
# MAIN
# =========================
dem, profile, nodata, res_x, res_y = read_dem(dem_path)

if not np.isclose(res_x, res_y):
    raise ValueError(
        f"DEM has non-square pixels: res_x={res_x}, res_y={res_y}. "
        "Для RVT и анализа рельефа лучше использовать квадратный пиксель."
    )

# SLOPE
slope = compute_slope(dem, res_x, res_y, units=slope_units)

# SAVE
out_dir = Path(dem_path).resolve().parent
save_raster(str(out_dir / "slope21_opt.tif"), slope, profile)

print("Done.")
print(f"Input DEM: {dem_path}")
print(f"Resolution: {res_x} m/pixel")
print("Saved:")
print(f"  {out_dir / 'slope21_opt.tif'}")