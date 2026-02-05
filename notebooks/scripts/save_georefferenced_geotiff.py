import rasterio
import numpy as np
import copy
#import matplotlib.pyplot as plt
#import cv2

#srcPath = '../data/output/Unet_IOU_FP_to_show.npz'
#srcPath = '../data/output/MRCNN_Output_Full.npz'
#origPath = '../data/DEM.tif'
#origPath = '/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/georef/DEM.tif'
origPath = '/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/data/QGIS/identifikacia stavieb/identifikacia stavieb/DEM21_opt.tif'
#trgtPath = '../data/output/MRCNN_Output_Full_to_show.tif'
#trgtPath = '../data/output/UNET_Output_to_show.tif'
#trgtPath = '../data/output/Buildings_only.tif'
#trgtPath = '../data/output/Looting_only.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_11_47.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_11_03.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_Agreg.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_Mounds_Agreg.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_Platforms.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_Mounds_Miro.tif'
#trgtPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/DEM21_opt_Structures_Miro.tif'

#src = np.load(srcPath)
#data = src['GT']
#data = cv2.imread('../data/output/Test_IOU_based_box30_valid.png')
#data = cv2.imread('../data/output/mask_pure_structures.tif')
#data = cv2.imread('../data/output/looting_mask.tif')

#pngPath = 'c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff//Remake s Mirom/prediction_mounds_10_59_05_05_2022.png'
pngPath = '/home/nikitachernysh/storage/Projects/lidar-archaeology-segmentation/notebooks/0.80(5)(Pretrained)_DEM21_opt_map.png'

trgtPath = pngPath[:-4] + '.tif'
#trgtPath = 'ph2' + '.tif'

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
data = np.array(Image.open(pngPath))
#data = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_11_03.png'))
#data = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_platforms.png'))
#data = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_mounds_Miro.png'))
#data = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_structures_Miro.png'))

# For aggregated mask
#data = (data>0).astype(np.uint8)
#dt = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_11_03.png'))
#data += (dt>0).astype(np.uint8)
#dt = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_mounds_Miro.png'))
#data += (dt>0).astype(np.uint8)
#del dt


# For aggregated mask
#data = (data>0).astype(np.uint8)
#dt = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_mounds_Miro.png'))
#data += (dt>0).astype(np.uint8)
#dt = np.array(Image.open('c:/Users/marek/Downloads/Guatemala/Nové dáta/GeoTiff/prediction_structures_Miro.png'))
#data += (dt>0).astype(np.uint8)
#del dt

data = (data>0).astype(np.uint8)
#data = (data*0).astype(np.uint8)


#src = rasterio.open(origPath)
#dataset = src.read(1,masked = False)
#data = data.astype('float64') 
#  
#data = data.astype('uint8') 
print(data.dtype)
print(data.shape)
print(np.min(data))
print(np.max(data))
#
##data = (data/255).astype('uint8')
#data = (data[:,:,0] + data[:,:,1] +data[:,:,2])
#print(np.min(data))
#print(np.max(data))
#data = (data/255/3).astype('uint8')

#data = (data[:,:,0]*1 + data[:,:,1]*2 +data[:,:,2]*3)
## 0 = TN, 1 = TP, 2 = FP, 3 = FN (zistit, preco su F naopak)
#data = np.zeros(data.shape,dtype = 'uint8') + (data == 6)*1 + (data == 3)*2 + (data == 1)*3

#print(data.shape)
#print(np.min(data))
#print(np.max(data))


#selection = src['validSelectionOrig']
# selection ordering (column min, row min, column max, row max)
selection = [0, 0, data.shape[1], data.shape[0]]
#selection = (442, 1107, 9177, 5490)

orig = rasterio.open(origPath)
#count = orig.count,
#width = orig.width,
#height = orig.height,
#indexes = orig.indexes, 
#dtypes = orig.dtypes,
bounds = orig.bounds,
transform = orig.transform,
crs = orig.crs

origTransform = copy.deepcopy(transform)

#origImage = orig.read(1,masked = True)
#print(origImage.shape)
#print(width[0])
#print(height[0])

#print(bounds)
#print(orig.transform * (0,0))
#print(orig.transform * (width[0],height[0]))

[left,top] = orig.transform * (selection[0],selection[1])
#print(left)
#print(top)
[right,bottom] = orig.transform * (selection[2],selection[3])
#print(right)
#print(bottom)
#print(data.shape)

print(origTransform)
newTransform = rasterio.transform.Affine.translation(left, top)\
    * rasterio.transform.Affine.scale(origTransform[0][0], origTransform[0][4])
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

#trgt.write(data, 1)
trgt.write(data, 1)
#mask = ((data==0)*255).astype('uint8')
#plt.imshow(mask)
#trgt.write_mask(mask)

trgt.close()
orig.close()

#
#src = rasterio.open(srcPath)
#dataset = src.read(1,masked = False)
#validMask = src.read_masks(1)
#src.close()
#
#np.savez_compressed(trgtPath, dataset=dataset, validMask=validMask)
#
#loaded = np.load(trgtPath + ".npz")
#print(np.array_equal(dataset, loaded['dataset']))
#print(np.array_equal(validMask, loaded['validMask']))

