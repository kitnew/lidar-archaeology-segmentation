import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
import rvt.vis


# =========================
# INPUT
# =========================
#dem_path = "/home/nikitachernysh/Storage/Projects/lidar-archaeology-segmentation/data/georef/DEM.tif"
dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM21_opt.tif"

# Радиус анализа в метрах.
# Подбери под масштаб объектов:
# 5-10 м -> мелкие объекты
# 10-30 м -> более крупные формы
radius_m = 10.0

# Количество направлений для SVF / PO
svf_n_dir = 16

# Подавление шума:
# 0 - без подавления
# 1 - low
# 2 - medium
# 3 - high
svf_noise = 0

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

# =========================
# MAIN
# =========================
dem, profile, nodata, res_x, res_y = read_dem(dem_path)

if not np.isclose(res_x, res_y):
    raise ValueError(
        f"DEM has non-square pixels: res_x={res_x}, res_y={res_y}. "
        "Для RVT и анализа рельефа лучше использовать квадратный пиксель."
    )

# RVT ожидает радиус в пикселях
radius_px = max(1, int(round(radius_m / res_x)))

# 1) SVF + POSITIVE OPENNESS
# В RVT SVF, anisotropic SVF и positive openness считаются одной функцией.
svf_dict = rvt.vis.sky_view_factor(
    dem=dem,
    resolution=res_x,
    compute_svf=True,
    compute_asvf=False,
    compute_opns=True,
    svf_n_dir=svf_n_dir,
    svf_r_max=radius_px,
    svf_noise=svf_noise,
    no_data=np.nan
)

svf = svf_dict["svf"].astype(np.float32)
po = svf_dict["opns"].astype(np.float32)

# 2) SAVE
out_dir = Path(dem_path).resolve().parent
save_raster(str(out_dir / f"svf21_{radius_px}.tif"), svf, profile)
save_raster(str(out_dir / f"positive_openness21_{radius_px}.tif"), po, profile)

print("Done.")
print(f"Input DEM: {dem_path}")
print(f"Resolution: {res_x} m/pixel")
print(f"Radius: {radius_m} m -> {radius_px} px")
print("Saved:")
print(f"  {out_dir / f'svf21_{radius_px}.tif'}")
print(f"  {out_dir / f'positive_openness21_{radius_px}.tif'}")