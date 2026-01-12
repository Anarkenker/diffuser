import open3d as o3d
import numpy as np

sample_path = "arkit_demo.npy"

def load_xyz_labels(path: str):
    """
    返回 xyz (N,3) 和 labels (N,) 或 None。
    兼容最常见的 (N,4) numpy 保存格式。
    """
    try:
        data = np.load(path, allow_pickle=False)
    except ValueError:
        data = np.load(path, allow_pickle=True)

    # 普通二维数组
    if isinstance(data, np.ndarray) and data.ndim == 2:
        xyz = data[:, :3].astype(np.float64, copy=False)
        labels = data[:, 3].astype(np.int32, copy=False) if data.shape[1] >= 4 else None
        return xyz, labels

    # 结构化数组
    if isinstance(data, np.ndarray) and data.dtype.fields is not None:
        fields = data.dtype.fields
        if all(k in fields for k in ("x", "y", "z")):
            xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64, copy=False)
            lbl = None
            for k in ("label", "seg", "cls"):
                if k in fields:
                    lbl = data[k].astype(np.int32, copy=False)
                    break
            return xyz, lbl

    raise ValueError(f"Unsupported npy content type/shape: type={type(data)}, shape={getattr(data,'shape',None)}")


xyz, labels = load_xyz_labels(sample_path)

# （可选）下采样，点太多会卡
max_points = 500_000
if xyz.shape[0] > max_points:
    idx = np.random.choice(xyz.shape[0], max_points, replace=False)
    xyz = xyz[idx]
    if labels is not None:
        labels = labels[idx]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# 给标签着色（背景黑，前景红；未知灰）
if labels is not None:
    palette = np.array([[0, 0, 0], [255, 0, 0], [128, 128, 128]], dtype=np.float64) / 255.0
    lbl = labels.copy()
    lbl[lbl < 0] = 2
    lbl[lbl >= palette.shape[0]] = 2
    colors = palette[lbl]
    pcd.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="ARKit PointCloud")
vis.add_geometry(pcd)

opt = vis.get_render_option()
opt.background_color = np.array([1.0, 1.0, 1.0])
opt.point_size = 3.0
opt.show_coordinate_frame = False

vis.run()
vis.destroy_window()
