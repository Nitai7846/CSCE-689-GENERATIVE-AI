#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 18:33:27 2025

@author: nitaishah
"""

import numpy as np
from pathlib import Path


# =============== USER INPUTS ===============
K_PATH = Path("/Volumes/One Touch/GEN_AI_PROJECT/phase_3/DA3/HPRC_OUTPUTS/cameras/pano_1_yaw30_pitch0_K.npy")   # update if needed
E_PATH = Path("/Volumes/One Touch/GEN_AI_PROJECT/phase_3/DA3/HPRC_OUTPUTS/cameras/pano_1_yaw30_pitch0_E.npy")   # update if needed
# ==========================================


def summarize_basic(name, arr):
    print(f"\n========== {name} ==========")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print("Values:\n", arr)
    print("Min / Max:", float(arr.min()), "/", float(arr.max()))
    print("Mean / Std:", float(arr.mean()), "/", float(arr.std()))


def analyze_K(K):
    """
    Expecting K as a 3x3 intrinsic matrix:
        [ fx   0  cx ]
        [  0  fy  cy ]
        [  0   0   1 ]
    """
    summarize_basic("K (intrinsics)", K)

    if K.shape != (3, 3):
        print("\n[WARN] K is not 3x3, unexpected for a standard pinhole camera model.")
        return

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    print("\n--- Interpreted intrinsics ---")
    print(f"fx: {fx}")
    print(f"fy: {fy}")
    print(f"cx: {cx}")
    print(f"cy: {cy}")

    # Basic sanity checks
    if fx <= 0 or fy <= 0:
        print("[WARN] fx or fy is non-positive. That is unusual for a real camera.")
    if abs(K[0, 1]) > 1e-6 or abs(K[1, 0]) > 1e-6:
        print("[NOTE] Off-diagonal focal terms are non-zero (skewed intrinsics or non-standard form).")
    if abs(K[2, 0]) > 1e-6 or abs(K[2, 1]) > 1e-6 or abs(K[2, 2] - 1.0) > 1e-6:
        print("[NOTE] Last row is not [0, 0, 1] – check if K is in a different convention.")


def analyze_rotation(R):
    """
    Check if R is approximately a valid rotation matrix:
    R^T R ≈ I, det(R) ≈ 1.
    """
    print("\n--- Rotation matrix diagnostics ---")
    if R.shape != (3, 3):
        print(f"[WARN] R is not 3x3, got {R.shape}")
        return

    RtR = R.T @ R
    I = np.eye(3)
    ortho_error = np.linalg.norm(RtR - I)
    detR = np.linalg.det(R)

    print("R:\n", R)
    print("R^T R:\n", RtR)
    print("‖R^T R - I‖_F:", ortho_error)
    print("det(R):", detR)

    if ortho_error < 1e-3 and abs(detR - 1.0) < 1e-3:
        print("[OK] R looks like a valid rotation matrix.")
    else:
        print("[WARN] R does NOT look like a clean rotation matrix (numerical issues or wrong pose?).")


def analyze_E(E):
    """
    E is expected to be either:
    - 3x4: [R | t]
    - 4x4: [R | t; 0 0 0 1]
    """
    summarize_basic("E (extrinsics / pose)", E)

    if E.shape == (3, 4):
        print("\n[INFO] E is 3x4, interpreting as [R | t] world→camera.")
        R = E[:, :3]
        t = E[:, 3].reshape(3, 1)

    elif E.shape == (4, 4):
        print("\n[INFO] E is 4x4, homogeneous transform.")
        R = E[:3, :3]
        t = E[:3, 3].reshape(3, 1)
        last_row = E[3, :]
        print("Last row of E:", last_row)
        if not np.allclose(last_row, np.array([0, 0, 0, 1]), atol=1e-6):
            print("[WARN] Last row not [0, 0, 0, 1]. This is unusual for a standard SE(3) transform.")
    else:
        print(f"\n[WARN] Unexpected E shape {E.shape}. Cannot interpret as standard extrinsics.")
        return

    print("\n--- Decomposed extrinsics ---")
    print("R (3x3):\n", R)
    print("t (3x1):\n", t.ravel())

    analyze_rotation(R)

    # If E is world->camera: X_c = R X_w + t
    # Then camera center in world coordinates is: C = -R^T t
    C_world = -R.T @ t
    print("\nEstimated camera center (assuming E is world→camera):")
    print("C_world:", C_world.ravel())


def main():
    print("Loading K and E from disk...")
    K = np.load(K_PATH)
    E = np.load(E_PATH)

    analyze_K(K)
    analyze_E(E)


if __name__ == "__main__":
    main()