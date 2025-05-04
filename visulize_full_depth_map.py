import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the depth map
depth_map_path = r"full_depth_maps\DJI_20241123144724_0040_V.npy"
depth_map = np.load(depth_map_path)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Display raw depth values
im1 = ax1.imshow(depth_map, cmap='viridis')
ax1.set_title('Raw Depth Map')
fig.colorbar(im1, ax=ax1, label='Depth (meters)')

# Display normalized depth for better visualization
valid_depths = depth_map[depth_map > 0]  # Exclude background (0 values)
if len(valid_depths) > 0:
    vmin, vmax = np.percentile(valid_depths, [1, 99])
else:
    vmin, vmax = 0, 1

im2 = ax2.imshow(depth_map, cmap='plasma', vmin=vmin, vmax=vmax)
ax2.set_title('Normalized Depth (percentile)')
fig.colorbar(im2, ax=ax2, label='Depth (meters)')

# Add statistics
print(f"Depth Map Statistics:")
print(f"Size: {depth_map.shape} (H x W)")
print(f"Min depth: {np.min(valid_depths) if len(valid_depths) > 0 else 'N/A'}")
print(f"Max depth: {np.max(valid_depths) if len(valid_depths) > 0 else 'N/A'}")
print(f"Mean depth: {np.mean(valid_depths) if len(valid_depths) > 0 else 'N/A'}")
print(f"Background pixels: {np.sum(depth_map == 0)/depth_map.size:.1%}")

plt.tight_layout()
plt.show()

# Optional: Create 3D visualization
# try:
#     from mpl_toolkits.mplot3d import Axes3D
    
#     # Sample every 10th pixel for faster 3D plotting
#     h, w = depth_map.shape
#     y, x = np.mgrid[0:h:10, 0:w:10]
#     z = depth_map[::10, ::10]
    
#     fig3d = plt.figure(figsize=(10, 8))
#     ax3d = fig3d.add_subplot(111, projection='3d')
#     surf = ax3d.plot_surface(x, y, z, cmap='viridis',
#                            linewidth=0, antialiased=False)
#     ax3d.set_title('3D Depth Visualization')
#     ax3d.set_xlabel('Width (pixels)')
#     ax3d.set_ylabel('Height (pixels)')
#     ax3d.set_zlabel('Depth (meters)')
#     fig3d.colorbar(surf, shrink=0.5, aspect=5)
#     plt.show()
# except ImportError:
#     print("3D visualization requires mpl_toolkits")