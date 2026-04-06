import numpy as np

dem_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/DEM_normalized.npz"
mask_path = "/home/nc225mj/lidar-archaeology-segmentation/data/processed/mounds_mask_shadowed.npy"

data = np.load(dem_path)
dem = data["dem"].astype(np.float32)
valid = data["valid"].astype(bool)
mask = np.load(mask_path).astype(np.uint8)

from skimage.measure import label, regionprops

labels = label(mask)
props = regionprops(labels)

areas = [region.area for region in props]

avg_area = np.mean(areas)
print(f"Average area: {avg_area}")
print(f"Min area: {np.min(areas)}")
print(f"Max area: {np.max(areas)}")

largest_region = max(props, key=lambda region: region.area)

import matplotlib.pyplot as plt

# Get the bounding box of the largest region
min_row, min_col, max_row, max_col = largest_region.bbox
padding = 10  # Add some padding around the region

# Calculate coordinates with padding, ensuring they're within image bounds
min_row = max(0, min_row - padding)
min_col = max(0, min_col - padding)
max_row = min(mask.shape[0], max_row + padding)
max_col = min(mask.shape[1], max_col + padding)

# Extract the region from the original mask
region = mask[min_row:max_row, min_col:max_col]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original mask
ax1.imshow(mask, cmap='gray')
ax1.set_title('Original Mask')

# Plot the region with the largest component highlighted
ax2.imshow(region, cmap='gray')
ax2.set_title(f'Largest Region (Area: {largest_region.area} pixels)')

# Add a rectangle around the region in the original mask
rect = plt.Rectangle((min_col, min_row), 
                    max_col - min_col, 
                    max_row - min_row,
                    fill=False, edgecolor='red', linewidth=2)
ax1.add_patch(rect)

plt.tight_layout()
plt.savefig("largest_region.png")