import rasterio
from rasterio.windows import from_bounds
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches

# --- ПУТИ ---
hillshade_path = "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/identifikacia stavieb/identifikacia stavieb/DEM21_opt_Hillshade.tif"
labels_path = "/home/nc225mj/lidar-archaeology-segmentation/data/QGIS/identifikacia stavieb/identifikacia stavieb/U2_Uaxactun_structures.geojson"

# 1. Загрузка данных
print(-1)
labels = gpd.read_file(labels_path)
print(0)
with rasterio.open(hillshade_path) as src:
    full_hs = src.read(1).astype('float32')
    full_transform = src.transform
    crs = src.crs
    nodata = src.nodata
    full_hs[full_hs == nodata] = np.nan
print(1)
# Определяем границы аннотированной области (AOI)
# Добавим небольшой отступ (padding) 50 метров, чтобы рамка не прилипала к структурам
minx, miny, maxx, maxy = labels.total_bounds
pad = 50
aoi_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

# Загружаем только фрагмент для панели (B), чтобы не тратить память
with rasterio.open(hillshade_path) as src:
    window = from_bounds(*aoi_bounds, src.transform)
    aoi_hs = src.read(1, window=window).astype('float32')
    aoi_transform = src.window_transform(window)
    aoi_hs[aoi_hs == nodata] = np.nan
print(2)
# 2. Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

# --- PANEL (A): OVERVIEW ---
vmin1, vmax1 = np.nanpercentile(full_hs, [2, 98])
ax1.imshow(full_hs, cmap='gray', vmin=vmin1, vmax=vmax1, extent=[
    full_transform[2], full_transform[2] + full_transform[0] * full_hs.shape[1],
    full_transform[5] + full_transform[4] * full_hs.shape[0], full_transform[5]
])
print(3)
# Рисуем рамку AOI
rect = mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                          linewidth=2, edgecolor='red', facecolor='none', label='Annotated Area')
ax1.add_patch(rect)
ax1.set_title("(a) Full Scan Overview", loc='left', fontsize=14, fontweight='bold')
ax1.set_axis_off()
print(4)
# Добавим стрелку севера только на обзорную карту
ax1.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.87),
             arrowprops=dict(facecolor='black', width=1, headwidth=8),
             xycoords='axes fraction', ha='center', va='center', fontsize=12)

# --- PANEL (B): DETAILED VIEW ---
vmin2, vmax2 = np.nanpercentile(aoi_hs, [2, 98])
ax2.imshow(aoi_hs, cmap='gray', vmin=vmin2, vmax=vmax2, extent=[
    aoi_transform[2], aoi_transform[2] + aoi_transform[0] * aoi_hs.shape[1],
    aoi_transform[5] + aoi_transform[4] * aoi_hs.shape[0], aoi_transform[5]
])
print(5)
# Накладываем labels
labels.plot(ax=ax2, color='cyan', alpha=0.4, edgecolor='cyan', linewidth=0.5)

ax2.set_title("(b) Annotated Area Detail", loc='left', fontsize=14, fontweight='bold')
ax2.set_axis_off()

# Масштабная линейка для детальной области
scalebar = ScaleBar(1, "m", location="lower right", frameon=True, box_alpha=0.7)
ax2.add_artist(scalebar)
print(7)
plt.tight_layout()
plt.savefig('maya.png', dpi=300, bbox_inches='tight')