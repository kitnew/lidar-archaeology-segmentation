from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter
import rasterio
import copy

origPath = [
    '/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM_JZ_1m.tiff',
    '/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM_JZ_1m.tiff',
    '/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM_MC_1m.tiff',
    '/home/nc225mj/lidar-archaeology-segmentation/data/georef/DEM_MC_1m.tiff'
]
preds = [
    "/home/nc225mj/lidar-archaeology-segmentation/experiments/JZ_Pretrained/JZ_full_prediction_map.npy",
    "/home/nc225mj/lidar-archaeology-segmentation/experiments/JZ/JZ_full_prediction_map.npy",
    "/home/nc225mj/lidar-archaeology-segmentation/experiments/MC_Pretrained/MC_full_prediction_map.npy",
    "/home/nc225mj/lidar-archaeology-segmentation/experiments/MC/MC_full_prediction_map.npy",
]
b_threshold = [
    0.72,
    0.72,
    0.72,
    0.72
]

for path, b_th, orig_path in zip(preds, b_threshold, origPath):
    preds = np.load(path)

    smoothed = uniform_filter(preds, size=5)

    # binarize
    preds = (smoothed > b_th).astype(np.uint8)

    # save to uint8
    img = (preds * 255).astype(np.uint8)

    #Image.fromarray(img).save("DEM21_opt_map.png")

    #pngPath = '/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/notebooks/0.80(5)(Pretrained)_DEM21_opt_map.npz'

    trgtPath = path[:-4] + '.tif'

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    data = img

    data = (data>0).astype(np.uint8)

    print(data.dtype)
    print(data.shape)

    selection = [0, 0, data.shape[1], data.shape[0]]

    orig = rasterio.open(orig_path)
    bounds = orig.bounds
    transform = orig.transform
    crs = orig.crs

    origTransform = copy.deepcopy(transform)

    [left,top] = orig.transform * (selection[0],selection[1])
    [right,bottom] = orig.transform * (selection[2],selection[3])

    print(origTransform)
    newTransform = rasterio.transform.Affine.translation(left, top)\
        * rasterio.transform.Affine.scale(origTransform.a, origTransform.e)
    print(newTransform)

    trgt = rasterio.open(
            trgtPath,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=newTransform,
            nodata = 255
            )

    trgt.write(data, 1)

    trgt.close()
    orig.close()
