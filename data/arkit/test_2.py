import numpy as np, open3d as o3d
pcd = o3d.io.read_point_cloud("arkit_demo.ply")
colors = (np.asarray(pcd.colors) * 255).round().astype(int)
uniq_colors, counts = np.unique(colors, axis=0, return_counts=True)
print("颜色及数量:", list(zip(map(tuple, uniq_colors), counts)))
