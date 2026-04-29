import numpy as np
from scipy.ndimage import uniform_filter
import rasterio
import copy
import os

origPath = [
    "/home/nc225mj/lidar-archaeology-segmentation/data/georef/Hradisko_Tribec_DEM25cm.tif"
]
preds_paths = [
    "/home/nc225mj/lidar-archaeology-segmentation/outputs/inference/2026-04-20/12-22-44/full_prediction_map.npy"
]
b_threshold = [
    0.208
]

for path, b_th, orig_path in zip(preds_paths, b_threshold, origPath):
    # Load raw predictions (after sigmoid)
    raw_preds = np.load(path)
    
    # Apply smoothing
    smoothed = uniform_filter(raw_preds, size=5)

    # 1. Prepare Binary Map
    binary_data = (smoothed > b_th).astype(np.uint8)

    # 2. Prepare Probability Map (smoothed raw values)
    prob_data = smoothed.astype(np.float32)

    # Georeferencing setup
    orig = rasterio.open(orig_path)
    crs = orig.crs
    transform = orig.transform
    
    # Define target paths
    base_name = os.path.splitext(path)[0]
    binary_trgtPath = base_name + '_binary.tif'
    prob_trgtPath = base_name + '_prob.tif'

    # Common metadata for writing
    meta = {
        'driver': 'GTiff',
        'height': binary_data.shape[0],
        'width': binary_data.shape[1],
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }

    # Save Binary Map
    binary_meta = meta.copy()
    binary_meta.update(dtype='uint8', nodata=255)
    with rasterio.open(binary_trgtPath, 'w', **binary_meta) as dst:
        dst.write(binary_data, 1)
    print(f"Saved binary map to: {binary_trgtPath}")

    # Save Probability Map
    prob_meta = meta.copy()
    prob_meta.update(dtype='float32', nodata=-1.0) # Using -1.0 for float nodata
    with rasterio.open(prob_trgtPath, 'w', **prob_meta) as dst:
        dst.write(prob_data, 1)
    print(f"Saved probability map to: {prob_trgtPath}")

    orig.close()
