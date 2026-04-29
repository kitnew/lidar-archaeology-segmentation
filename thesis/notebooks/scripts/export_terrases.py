import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

FILL_VALUE = np.nan

AOIS = [
    {
        "name": "Ladzany",
        "split": "train",
        "dem": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Ladzany_DEM25cm.tif",
        "iso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Ladzany_VIS_ISO.tif",
        "aniso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Ladzany_VIS_ANISO.tif",
        "labels": "/home/nc225mj/lidar-archaeology-segmentation/data/raw/Ladzany_mask.gpkg",
    },
    {
        "name": "Sasovske",
        "split": "train",
        "dem": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Sasovske_Podhradie_DEM25cm.tif",
        "iso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Sasovske_Podhradie_VIS_ISO.tif",
        "aniso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Sasovske_Podhradie_VIS_ANISO.tif",
        "labels": "/home/nc225mj/lidar-archaeology-segmentation/data/raw/Sasovske_mask.gpkg",
    },
    {
        "name": "Sitno",
        "split": "train",
        "dem": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Sitno_DEM25cm.tif",
        "iso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Sitno_VIS_ISO.tif",
        "aniso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Sitno_VIS_ANISO.tif",
        "labels": "/home/nc225mj/lidar-archaeology-segmentation/data/raw/Sitno_mask.gpkg",
    },
    {
        "name": "Tribec",
        "split": "test",
        "dem": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Tribec_DEM25cm.tif",
        "iso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Tribec_VIS_ISO.tif",
        "aniso": "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/250227_Bundzel_Sperka_Terasy/Hradisko_Tribec_VIS_ANISO.tif",
        "labels": None,
    },
]

OUT_DIR = Path("packed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)   # always (B, H, W)
        meta = {
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "bounds": src.bounds,
        }
    return arr, meta


def assert_same_grid(ref_meta, other_meta, name):
    assert ref_meta["height"] == other_meta["height"], f"{name}: height mismatch"
    assert ref_meta["width"] == other_meta["width"], f"{name}: width mismatch"
    assert ref_meta["transform"] == other_meta["transform"], f"{name}: transform mismatch"
    assert ref_meta["crs"] == other_meta["crs"], f"{name}: crs mismatch"


def build_valid_mask(arrs, nodatas):
    H, W = arrs[0].shape[-2:]
    valid = np.ones((H, W), dtype=bool)

    for arr, nodata in zip(arrs, nodatas):
        valid &= np.all(np.isfinite(arr), axis=0)
        if nodata is not None:
            valid &= np.all(arr != nodata, axis=0)

    return valid.astype(np.uint8)


def rasterize_labels(labels_path, ref_meta):
    if labels_path is None:
        return np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.uint8), np.uint8(0)

    gdf = gpd.read_file(labels_path)

    if gdf.empty:
        return np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.uint8), np.uint8(1)

    gdf = gdf.to_crs(ref_meta["crs"])

    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]

    mask = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=(ref_meta["height"], ref_meta["width"]),
        transform=ref_meta["transform"],
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    return mask, np.uint8(1)


for aoi in AOIS:
    dem, dem_meta = read_raster(aoi["dem"])       # (1, H, W)
    iso, iso_meta = read_raster(aoi["iso"])       # (3, H, W)
    aniso, aniso_meta = read_raster(aoi["aniso"]) # (3, H, W)

    assert dem.shape[0] == 1, f'{aoi["name"]}: DEM must be single-band, got {dem.shape}'
    assert iso.shape[0] in (3, 4), f'{aoi["name"]}: ISO must have 3 or 4 bands, got {iso.shape}'
    assert aniso.shape[0] in (3, 4), f'{aoi["name"]}: ANISO must have 3 or 4 bands, got {aniso.shape}'
    
    if iso.shape[0] == 4:
        iso = iso[:3]
    
    if aniso.shape[0] == 4:
        aniso = aniso[:3]

    assert_same_grid(dem_meta, iso_meta, f'{aoi["name"]}: ISO')
    assert_same_grid(dem_meta, aniso_meta, f'{aoi["name"]}: ANISO')

    mask, has_labels = rasterize_labels(aoi["labels"], dem_meta)

    valid = build_valid_mask(
        [dem, iso, aniso],
        [dem_meta["nodata"], iso_meta["nodata"], aniso_meta["nodata"]],
    )

    data = np.concatenate([dem, iso, aniso], axis=0).astype(np.float32)   # (7, H, W)
    data[:, valid == 0] = FILL_VALUE

    if has_labels:
        mask[valid == 0] = 0

    meta = {
        "name": aoi["name"],
        "split": aoi["split"],
        "channels": ["DEM", "ISO_R", "ISO_G", "ISO_B", "ANISO_R", "ANISO_G", "ANISO_B"],
        "height": dem_meta["height"],
        "width": dem_meta["width"],
        "crs": str(dem_meta["crs"]),
        "transform": tuple(dem_meta["transform"]),
        "fill_value": None if np.isnan(FILL_VALUE) else FILL_VALUE,
    }

    out_path = OUT_DIR / f'{aoi["name"]}.npz'
    np.savez_compressed(
        out_path,
        data=data,
        valid=valid.astype(np.uint8),
        mask=mask.astype(np.uint8),
        has_labels=has_labels,
        meta=json.dumps(meta),
    )

    print(f"Saved: {out_path}")
    print("  data:", data.shape, data.dtype)
    print("  mask:", mask.shape, mask.dtype, "positive:", int(mask.sum()))
    print("  valid:", valid.shape, valid.dtype, "valid pixels:", int(valid.sum()))