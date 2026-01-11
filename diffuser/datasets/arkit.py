# Copyright (c) 2021, ETHZ V4RL. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# ARKit dataset adapter (single-scene, 1-N frames)
#
# Expected layout (single frame):
#   data_root/
#     cloud.ply                # optional but preferred; otherwise points.npy / points_f32.bin
#     points.npy | points_f32.bin         # optional fallback for point cloud
#     normals.npy | normals_f32.bin       # optional normals aligned with points
#     rgb.jpg                  # optional (Diffuser does not use RGB)
#     depth.npy | depth_f32.bin
#     labels.png               # integer id map; binary mask 0/1 is fine
#     meta.json                # intrinsics + pose + resolutions (see README below)
#
# Expected layout (multi-frame):
#   data_root/
#     cloud.ply  (or points.npy / points_f32.bin)
#     frames/000000/{rgb.jpg, depth.npy, labels.png, meta.json}
#     frames/000001/{...}
#
# meta.json fields supported (case-insensitive):
#   K_rgb / intrinsics / K / camera_intrinsics : 3x3 matrix
#   K_depth                                    : 3x3 matrix (if missing, use K_rgb)
#   T_C_W                                      : 4x4 world->camera (preferred)
#   T_W_C / pose                               : 4x4 camera->world (will be inverted)
#   depth_resolution {w,h}                     : needed when depth is a raw .bin
#   unit ("meter"|"millimeter"|...)            : depth/points unit; meters assumed if absent
#
# Notes:
# - Labels are assumed to be integer IDs. If a binary mask is 0/255, it is normalized to {0,1}.
# - Diffuser only uses labels/depth/intrinsics/extrinsics/point cloud; RGB is ignored.

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from PIL import Image

from ..utils import get_sorted_file_list
from .base_dataset import BaseLabelTaxonomy, BaseScene
from .builder import register_dataset, register_label_taxonomy


# ----------------- helpers -----------------
def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_np(a, dtype=np.float32) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


def _mat33_from_meta(meta: Dict, keys: List[str]) -> np.ndarray:
    found = None
    for k in keys:
        for kk in meta.keys():
            if kk == k or kk.lower() == k.lower():
                found = meta[kk]
                break
        if found is not None:
            break
    if found is None:
        raise KeyError(f"meta.json missing intrinsics (any of {keys})")

    if isinstance(found, dict):
        fx = float(found.get("fx"))
        fy = float(found.get("fy"))
        cx = float(found.get("cx"))
        cy = float(found.get("cy"))
        s = float(found.get("skew", 0.0))
        return np.array([[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    arr = _as_np(found)
    if arr.shape == (3, 3):
        return arr.astype(np.float32)
    if arr.size == 9:
        return arr.reshape(3, 3).astype(np.float32)
    raise ValueError(f"Unsupported intrinsics format: shape={arr.shape}, size={arr.size}")


def _mat44_from_meta(meta: Dict) -> np.ndarray:
    # Prefer T_C_W / world->camera
    keys_cw = ["T_C_W", "T_camera_from_world", "world_to_camera", "extrinsics"]
    for k in keys_cw:
        for kk in meta.keys():
            if kk == k or kk.lower() == k.lower():
                T = _as_np(meta[kk])
                if T.shape == (4, 4):
                    return T.astype(np.float32)
                if T.size == 16:
                    return T.reshape(4, 4).astype(np.float32)
                raise ValueError(f"Unsupported T_C_W format: shape={T.shape}, size={T.size}")

    # Fallback: invert T_W_C / camera->world
    keys_wc = ["T_W_C", "T_world_from_camera", "camera_to_world", "pose"]
    for k in keys_wc:
        for kk in meta.keys():
            if kk == k or kk.lower() == k.lower():
                T_W_C = _as_np(meta[kk])
                if T_W_C.shape != (4, 4):
                    if T_W_C.size == 16:
                        T_W_C = T_W_C.reshape(4, 4)
                    else:
                        raise ValueError(
                            f"Unsupported T_W_C format: shape={T_W_C.shape}, size={T_W_C.size}"
                        )
                return np.linalg.inv(T_W_C.astype(np.float64)).astype(np.float32)

    raise KeyError("meta.json missing pose (T_C_W or T_W_C)")


def _pick_first_existing(root: str, names: List[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p):
            return p
    return None


def _discover_frames(data_root: str) -> Dict[str, List[str]]:
    """
    Returns dict with keys: depth, labels, meta (rgb optional).
    Paths are aligned by index.
    """
    frames_dir = os.path.join(data_root, "frames")
    if os.path.isdir(frames_dir):
        frame_dirs = sorted(
            [
                os.path.join(frames_dir, d)
                for d in os.listdir(frames_dir)
                if os.path.isdir(os.path.join(frames_dir, d))
            ]
        )
        depth, labels, meta = [], [], []
        for fd in frame_dirs:
            depth_path = _pick_first_existing(fd, ["depth.npy", "depth_f32.bin", "depth.png", "depth.pgm"])
            meta_path = _pick_first_existing(fd, ["meta.json", "frame.json", "info.json"])
            labels_path = _pick_first_existing(fd, ["labels.png", "label.png", "sem.png", "semantic.png"])
            if depth_path is None or meta_path is None:
                continue
            depth.append(depth_path)
            labels.append(labels_path if labels_path is not None else "")
            meta.append(meta_path)
        if len(depth) == 0:
            raise FileNotFoundError(f"No valid frames found under {frames_dir}")
        return {"depth": depth, "labels": labels, "meta": meta}

    # Single-frame fallback
    depth_path = _pick_first_existing(data_root, ["depth.npy", "depth_f32.bin", "depth.png", "depth.pgm"])
    meta_path = _pick_first_existing(data_root, ["meta.json", "frame.json", "info.json"])
    labels_path = _pick_first_existing(data_root, ["labels.png", "label.png", "sem.png", "semantic.png"])
    if depth_path is None or meta_path is None:
        raise FileNotFoundError(
            "Single-frame layout requires depth.(npy/bin/png/pgm) and meta.json in data_root."
        )
    return {"depth": [depth_path], "labels": [labels_path if labels_path else ""], "meta": [meta_path]}


def _read_depth(path: str, meta: Dict) -> np.ndarray:
    if path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)
    elif path.endswith(".bin"):
        if "depth_resolution" not in meta:
            raise KeyError("depth_resolution {w,h} required to read raw depth .bin")
        w = int(meta["depth_resolution"]["w"])
        h = int(meta["depth_resolution"]["h"])
        depth = np.fromfile(path, dtype=np.float32, count=w * h)
        depth = depth.reshape((h, w))
    else:
        depth = np.array(Image.open(path)).astype(np.float32)

    # Convert to meters if unit hints millimeters
    unit = meta.get("unit", "").lower()
    if unit and "mill" in unit:
        depth = depth / 1000.0
    else:
        # Heuristic fallback if not specified
        finite = depth[np.isfinite(depth)]
        if finite.size > 0 and np.nanmax(finite) > 50.0:
            depth = depth / 1000.0

    return depth


def _read_point_cloud(data_root: str) -> Tuple[o3d.geometry.PointCloud, Dict[str, str]]:
    info = {}
    ply_path = os.path.join(data_root, "cloud.ply")
    if os.path.isfile(ply_path):
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.has_points():
            info["source"] = ply_path
            return pcd, info

    # Fallback: npy/bin
    points_path = _pick_first_existing(data_root, ["points.npy", "points_f32.bin"])
    if points_path is None:
        raise FileNotFoundError("Point cloud not found (cloud.ply or points.npy / points_f32.bin)")
    if points_path.endswith(".npy"):
        points = np.load(points_path).astype(np.float32)
    else:
        points = np.fromfile(points_path, dtype=np.float32)
        if points.size % 3 != 0:
            raise ValueError(f"points_f32.bin size {points.size} not divisible by 3")
        points = points.reshape((-1, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    info["source"] = points_path

    normals_path = _pick_first_existing(data_root, ["normals.npy", "normals_f32.bin"])
    if normals_path is not None:
        if normals_path.endswith(".npy"):
            normals = np.load(normals_path).astype(np.float32)
        else:
            normals = np.fromfile(normals_path, dtype=np.float32)
            normals = normals.reshape((-1, 3))
        if normals.shape[0] == points.shape[0]:
            pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
            info["normals_source"] = normals_path

    return pcd, info


# ----------------- taxonomy -----------------
@register_label_taxonomy("arkit_binary")
class ARKitBinary(BaseLabelTaxonomy):
    CLASSES = ("background", "object")
    PALETTE = ((0, 0, 0), (255, 0, 0))

    def __init__(self):
        super().__init__()


# ----------------- dataset -----------------
@register_dataset("arkit")
class ARKitScene(BaseScene):
    def __init__(
        self,
        data_root: str,
        img_labels_dir: str = None,
        label_taxonomy: str = "arkit_binary",
        img_labels_suffix: str = ".png",
        img_labels_mapping=None,
    ):
        self.data_root = data_root
        self.frames_info = _discover_frames(data_root)
        self.label_taxonomy_name = label_taxonomy
        # If img_labels_dir is not provided, use data_root (BaseScene will call our override anyway)
        super().__init__(
            data_root,
            img_labels_dir or data_root,
            label_taxonomy,
            img_labels_suffix=img_labels_suffix,
            img_labels_mapping=img_labels_mapping,
        )
        self._pcd_info = {}
        self._T_cv_from_ca = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)  # ARKit cam -> OpenCV cam

    # ---- BaseScene required loaders ----
    def _load_img_intrinsics_info(self, data_root):
        # We keep meta paths for per-frame intrinsics
        return {"mode": "per_frame_meta", "meta_paths": self.frames_info["meta"]}

    def _load_img_extrinsics_info(self, data_root):
        return self.frames_info["meta"]

    def _load_img_depths_info(self, data_root):
        return self.frames_info["depth"]

    def _load_pcloud_info(self, data_root):
        # We override load_point_cloud, but still store a hint path for consistency.
        ply_path = os.path.join(data_root, "cloud.ply")
        if os.path.isfile(ply_path):
            return ply_path
        pts = _pick_first_existing(data_root, ["points.npy", "points_f32.bin"])
        return pts if pts is not None else ""

    def _load_img_labels_info(self, img_labels_dir, img_labels_suffix):
        # Align labels with discovered frames; fall back to img_labels_dir listing.
        labels = self.frames_info["labels"]
        if labels and any(os.path.isfile(p) for p in labels):
            return labels
        return get_sorted_file_list(img_labels_dir, file_ext=img_labels_suffix)

    # ---- frame loaders ----
    def load_img_depth(self, idx):
        meta = _read_json(self.img_extrinsics_info[idx])
        depth_path = self.img_depths_info[idx]
        depth = _read_depth(depth_path, meta)
        return depth

    def load_img_intrinsics(self, idx):
        meta = _read_json(self.img_intrinsics_info["meta_paths"][idx])
        K_rgb = _mat33_from_meta(meta, ["K_rgb", "intrinsics", "K", "camera_intrinsics", "color_intrinsics"])
        K_depth = _mat33_from_meta(meta, ["K_depth", "depth_intrinsics"]) if "K_depth" in meta or "depth_intrinsics" in meta else K_rgb
        return K_rgb, K_depth

    def load_img_extrinsics(self, idx):
        meta = _read_json(self.img_extrinsics_info[idx])
        T_C_W_arkit = _mat44_from_meta(meta)
        # Convert ARKit camera coords (x right, y up, z backward) -> OpenCV (x right, y down, z forward)
        T_C_W_cv = self._T_cv_from_ca @ T_C_W_arkit
        return T_C_W_cv

    def load_img_labels(self, idx):
        label_path = None
        if self.img_labels_info and len(self.img_labels_info) > idx:
            label_path = self.img_labels_info[idx]
        if not label_path or not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label image not found for idx={idx}")

        im = Image.open(label_path)
        if im.mode == "P":
            labels = np.array(im, dtype=np.int32)
        else:
            labels = np.array(im)
        if labels.ndim == 3:
            labels = labels[..., 0]
        labels = labels.astype(np.int32)

        if self.label_taxonomy_name == "arkit_binary":
            uniq = np.unique(labels)
            if uniq.size <= 3 and (255 in uniq or labels.max() > 1):
                labels = (labels > 0).astype(np.int32)
        return labels

    # ---- point cloud ----
    def load_point_cloud(self):
        pcd, info = _read_point_cloud(self.data_root)
        self._pcd_info = info
        return pcd
