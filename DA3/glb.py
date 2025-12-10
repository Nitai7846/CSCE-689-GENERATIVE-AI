#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 18:33:19 2025

@author: nitaishah
"""

import numpy as np
import trimesh
import open3d as o3d

# Path to your GLB
glb_path = "/Users/nitaishah/Downloads/scene.glb"

# 1. Load GLB as a trimesh.Scene
scene = trimesh.load(glb_path)
print(scene)

# 2. Extract the PointCloud from the scene
pc_tm = next(
    g for g in scene.geometry.values()
    if isinstance(g, trimesh.points.PointCloud)
)

points = np.asarray(pc_tm.vertices)   # (N, 3)
print("Points:", points.shape)

# 3. Create Open3D point cloud
pc_o3d = o3d.geometry.PointCloud()
pc_o3d.points = o3d.utility.Vector3dVector(points)

# Optional: handle colors if present
if hasattr(pc_tm, "colors") and pc_tm.colors is not None:
    colors = np.asarray(pc_tm.colors)  # usually uint8 RGBA
    if colors.shape[1] == 4:          # drop alpha if needed
        colors = colors[:, :3]
    colors = colors.astype(np.float64) / 255.0
    pc_o3d.colors = o3d.utility.Vector3dVector(colors)

# 4. Visualize
o3d.visualization.draw_geometries([pc_o3d])
