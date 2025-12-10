#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:17:32 2025

@author: nitaishah
"""

import os
import re
import open3d as o3d
from glob import glob


# -------------------------------------------------------------
# Extract yaw angle from filenames like: pano_1_yaw30_pitch0.ply
# -------------------------------------------------------------
def extract_yaw(filename):
    match = re.search(r"yaw(\d+)", filename)
    return int(match.group(1)) if match else None


# -------------------------------------------------------------
# Get all PLYs whose yaw matches yaw_filter list
# -------------------------------------------------------------
def load_plys_by_yaw(directory, yaw_filter):
    ply_files = sorted(glob(os.path.join(directory, "*.ply")))
    selected = []

    for f in ply_files:
        yaw = extract_yaw(f)
        if yaw is not None and yaw in yaw_filter:
            selected.append(f)

    print(f"\nSelected {len(selected)} PLY files for yaw(s) {yaw_filter}\n")
    return selected


# -------------------------------------------------------------
# Merge PLYs into a single point cloud
# -------------------------------------------------------------
def merge_pointclouds(file_list):
    merged = o3d.geometry.PointCloud()

    for f in file_list:
        print(f"Loading: {os.path.basename(f)}")
        pcd = o3d.io.read_point_cloud(f)

        # Ensure exact color preservation
        if not pcd.has_colors():
            print("Warning: PLY has no colors:", f)

        merged += pcd

    return merged


# -------------------------------------------------------------
# Main function: Merge + Show
# -------------------------------------------------------------
def merge_and_show(directory, yaw_list):
    files = load_plys_by_yaw(directory, yaw_list)
    if len(files) == 0:
        print("No files matched the yaw filter.")
        return

    merged_pcd = merge_pointclouds(files)

    print("\nDisplaying merged point cloud (close window to exit)...")

    # Viewer with correct colors
    o3d.visualization.draw_geometries(
        [merged_pcd],
        window_name="Merged Point Cloud — DepthAnything3",
        width=1280,
        height=960,
        left=100,
        top=100,
        point_show_normal=False
    )


# -------------------------------------------------------------
# Example usage (edit these before running)
# -------------------------------------------------------------
if __name__ == "__main__":

    # CHANGE THIS: Folder containing your DA3 pointclouds
    PLY_DIR = "/Volumes/One Touch/GEN_AI_PROJECT/phase_3/DA3/HPRC_OUTPUTS/pointclouds"

    # Choose yaw(s) to merge/view
    YAWS_TO_VIEW = [0]

    merge_and_show(PLY_DIR, YAWS_TO_VIEW)
