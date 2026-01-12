import numpy as np, json
from PIL import Image

meta = json.load(open("meta.json"))
K_rgb = np.array(meta["K_rgb"], dtype=np.float32)
K_depth = np.array(meta["K_depth"], dtype=np.float32)
T_C_W = np.array(meta["T_C_W"], dtype=np.float32)
pts = np.load("arkit_demo.npy")[:, :3]  # 或 cloud/points.npy
labels2d = np.array(Image.open("labels.png"))
depth = np.load("depth.npy").astype(np.float32)

# 投影到 RGB
pts_h = np.c_[pts, np.ones(len(pts))]
P_rgb = K_rgb @ T_C_W[:3, :]
proj = pts_h @ P_rgb.T
z = proj[:, 2]
px = np.column_stack([proj[:,0]/z + 0.5, proj[:,1]/z + 0.5]).astype(int)
in_img = (z > 0) & (px[:,0] >= 0) & (px[:,1] >= 0) & (px[:,0] < labels2d.shape[1]) & (px[:,1] < labels2d.shape[0])
print("可见点数(仅按RGB视锥):", in_img.sum())

# 深度一致性
P_d = K_depth @ T_C_W[:3, :]
proj_d = pts_h @ P_d.T
z_d = proj_d[:, 2]
px_d = np.column_stack([proj_d[:,0]/z_d + 0.5, proj_d[:,1]/z_d + 0.5]).astype(int)
in_depth = in_img & (px_d[:,0] >= 0) & (px_d[:,1] >= 0) & (px_d[:,0] < depth.shape[1]) & (px_d[:,1] < depth.shape[0]) & (z_d > 0)
mask = in_depth.copy()
th = 0.05  # 同 Diffuser
mask[in_depth] = np.abs(z_d[in_depth] - depth[px_d[in_depth,1], px_d[in_depth,0]]) < th
print("通过深度一致性的点数:", mask.sum())
