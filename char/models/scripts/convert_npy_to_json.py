#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSL .npy to 3D Motion JSON Pipeline
Converts raw Vietnamese Sign Language landmark arrays into clean, volumetric 3D JSON motions.
Strictly read-only on input workspace datasets.
Outputs written strictly to char/models/public/motions/ and char/models/public/models/.
"""

import os
import re
import sys
import json
import shutil
import urllib.request
import numpy as np

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def natural_sort_key(s):
    """Sort strings containing numbers in natural human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def ensure_vrm_model(models_dir, workspace_root):
    """Ensure AvatarSample_A.vrm exists in models_dir."""
    target_vrm = os.path.join(models_dir, "AvatarSample_A.vrm")
    if os.path.exists(target_vrm) and os.path.getsize(target_vrm) > 1000:
        print(f"[VRM] Verified AvatarSample_A.vrm exists ({os.path.getsize(target_vrm)} bytes)")
        return target_vrm

    # Search in workspace root (read-only)
    candidates = [
        os.path.join(workspace_root, "AvatarSample_A.vrm"),
        os.path.join(workspace_root, "avatar.vrm"),
        os.path.expanduser("~/Downloads/AvatarSample_A.vrm"),
        os.path.expanduser("~/Downloads/avatar.vrm")
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 1000:
            print(f"[VRM] Copying VRM model from candidate: {c}")
            shutil.copy2(c, target_vrm)
            return target_vrm

    # Fallback download
    url = "https://raw.githubusercontent.com/pixiv/three-vrm/v1.0.9/packages/three-vrm/examples/models/three-vrm-girl.vrm"
    print(f"[VRM] Downloading fallback VRM from {url}...")
    try:
        urllib.request.urlretrieve(url, target_vrm)
        print(f"[VRM] Downloaded fallback VRM model ({os.path.getsize(target_vrm)} bytes)")
        return target_vrm
    except Exception as e:
        print(f"[VRM Warning] Could not download fallback VRM: {e}")
        return None

def synthesize_3d_depth_pose(pose_2d):
    """
    Synthesize true biomechanical forward kinematic 3D depth for 8 pose points:
    0: Nose, 1: L_Shoulder, 2: R_Shoulder, 3: L_Elbow, 4: R_Elbow, 5: L_Wrist, 6: R_Wrist, 7: Mid_Shoulder
    Origin is Mid-Shoulder [0, 0, 0].
    MediaPipe depth convention: negative z is forward towards camera.
    """
    pose_3d = np.zeros((8, 3), dtype=np.float32)
    for i in range(8):
        pose_3d[i, 0] = pose_2d[i, 0]
        pose_3d[i, 1] = pose_2d[i, 1]

    if np.linalg.norm(pose_2d) < 1e-4:
        return pose_3d

    # Mid-shoulder origin
    pose_3d[7, 2] = 0.0
    # Left & Right shoulders (aligned at z = 0.0)
    pose_3d[1, 2] = 0.0
    pose_3d[2, 2] = 0.0
    # Nose slightly forward of shoulder plane
    pose_3d[0, 2] = -0.06

    L_upper = 0.95
    L_lower = 0.90

    # Left Arm forward kinematic depth
    if np.linalg.norm(pose_2d[5]) > 1e-4 or np.linalg.norm(pose_2d[3]) > 1e-4:
        d2D_upper = float(np.linalg.norm(pose_2d[3] - pose_2d[1]))
        d2D_lower = float(np.linalg.norm(pose_2d[5] - pose_2d[3]))
        is_raised = pose_2d[5, 1] < 0.6 or abs(pose_2d[5, 0]) < 0.65
        r_upper = min(1.0, max(0.0, d2D_upper / L_upper))
        r_lower = min(1.0, max(0.0, d2D_lower / L_lower))
        bend_upper = 1.0 - r_upper * r_upper
        bend_lower = 1.0 - r_lower * r_lower

        if is_raised:
            elbow_fwd = 0.06 + bend_upper * 0.14
            wrist_fwd = elbow_fwd + 0.08 + bend_lower * 0.14
            # Anti-torso clipping floor
            if abs(pose_2d[5, 0]) < 0.65 and pose_2d[5, 1] < 0.6:
                elbow_fwd = max(elbow_fwd, 0.14)
                wrist_fwd = max(wrist_fwd, 0.22)
            pose_3d[3, 2] = -elbow_fwd
            pose_3d[5, 2] = -wrist_fwd
        else:
            pose_3d[3, 2] = -0.03
            pose_3d[5, 2] = -0.05

    # Right Arm forward kinematic depth
    if np.linalg.norm(pose_2d[6]) > 1e-4 or np.linalg.norm(pose_2d[4]) > 1e-4:
        d2D_upper = float(np.linalg.norm(pose_2d[4] - pose_2d[2]))
        d2D_lower = float(np.linalg.norm(pose_2d[6] - pose_2d[4]))
        is_raised = pose_2d[6, 1] < 0.6 or abs(pose_2d[6, 0]) < 0.65
        r_upper = min(1.0, max(0.0, d2D_upper / L_upper))
        r_lower = min(1.0, max(0.0, d2D_lower / L_lower))
        bend_upper = 1.0 - r_upper * r_upper
        bend_lower = 1.0 - r_lower * r_lower

        if is_raised:
            elbow_fwd = 0.06 + bend_upper * 0.14
            wrist_fwd = elbow_fwd + 0.08 + bend_lower * 0.14
            # Anti-torso clipping floor
            if abs(pose_2d[6, 0]) < 0.65 and pose_2d[6, 1] < 0.6:
                elbow_fwd = max(elbow_fwd, 0.14)
                wrist_fwd = max(wrist_fwd, 0.22)
            pose_3d[4, 2] = -elbow_fwd
            pose_3d[6, 2] = -wrist_fwd
        else:
            pose_3d[4, 2] = -0.03
            pose_3d[6, 2] = -0.05

    return pose_3d

def synthesize_3d_depth_hand(hand_2d, wrist_z_base=-0.22):
    """
    Synthesize true volumetric 3D hand depth with palm transverse arch
    and forward circular arcs for curling fingers.
    """
    hand_3d = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        hand_3d[i, 0] = hand_2d[i, 0]
        hand_3d[i, 1] = hand_2d[i, 1]

    if np.linalg.norm(hand_2d) < 1e-4:
        return hand_3d

    hand_3d[0, 2] = wrist_z_base

    palm_len = float(np.linalg.norm(hand_2d[9, :2] - hand_2d[0, :2]))
    if palm_len < 1e-3:
        palm_len = 0.15

    scale_ratio = palm_len / 0.15

    # Palm transverse arch offsets relative to wrist
    mcp_arch_offsets = {
        1: -0.025 * scale_ratio,  # Thumb CMC
        2: -0.040 * scale_ratio,  # Thumb MCP
        5: -0.015 * scale_ratio,  # Index MCP
        9: -0.022 * scale_ratio,  # Middle MCP
        13: -0.015 * scale_ratio, # Ring MCP
        17: -0.005 * scale_ratio  # Pinky MCP
    }

    finger_chains = [
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
    ]

    for mcp, off in mcp_arch_offsets.items():
        hand_3d[mcp, 2] = wrist_z_base + off

    for chain in finger_chains:
        mcp, pip, dip, tip = chain
        direct = float(np.linalg.norm(hand_2d[tip, :2] - hand_2d[mcp, :2]))
        curve = max(
            float(np.linalg.norm(hand_2d[pip, :2] - hand_2d[mcp, :2]) +
                  np.linalg.norm(hand_2d[dip, :2] - hand_2d[pip, :2]) +
                  np.linalg.norm(hand_2d[tip, :2] - hand_2d[dip, :2])),
            0.001
        )
        curl = float(np.clip((0.85 - direct / curve) / (0.85 - 0.30), 0.0, 1.0))
        base_z = hand_3d[mcp, 2]

        hand_3d[pip, 2] = base_z - np.sin(curl * 1.2) * 0.025 * scale_ratio
        hand_3d[dip, 2] = base_z - np.sin(curl * 1.8) * 0.040 * scale_ratio
        hand_3d[tip, 2] = base_z - np.sin(curl * 2.4) * 0.035 * scale_ratio

    # Thumb IP & TIP
    hand_3d[3, 2] = hand_3d[2, 2] - 0.020 * scale_ratio
    hand_3d[4, 2] = hand_3d[2, 2] - 0.015 * scale_ratio

    return hand_3d

def process_single_frame(frame_data, dim_mode, dominant_side=None):
    """
    Process a single frame vector into 3D pose, leftHand, rightHand, and activeHandSide.
    """
    if dim_mode == 'one_hand_2d':
        pts = frame_data.reshape(29, 2)
        pose_2d = pts[0:8]
        hand_2d = pts[8:29]

        pose_3d = synthesize_3d_depth_pose(pose_2d)

        # Active hand side detection: Hand wrist (8) to L_Wrist (5) vs R_Wrist (6)
        if dominant_side is not None:
            active_side = dominant_side
        else:
            hand_wrist = hand_2d[0]
            l_wrist = pose_2d[5]
            r_wrist = pose_2d[6]
            d_l = np.linalg.norm(hand_wrist - l_wrist)
            d_r = np.linalg.norm(hand_wrist - r_wrist)
            active_side = 'leftHand' if d_l <= d_r else 'rightHand'

        if active_side == 'leftHand':
            wrist_z = pose_3d[5, 2]
            lh_3d = synthesize_3d_depth_hand(hand_2d, wrist_z_base=wrist_z)
            rh_3d = None
            # Inactive right arm hangs straight down below right shoulder
            pose_3d[4] = [pose_3d[2, 0], pose_3d[2, 1] + 0.95, 0.0]
            pose_3d[6] = [pose_3d[2, 0], pose_3d[2, 1] + 1.85, 0.0]
        else:
            wrist_z = pose_3d[6, 2]
            lh_3d = None
            rh_3d = synthesize_3d_depth_hand(hand_2d, wrist_z_base=wrist_z)
            # Inactive left arm hangs straight down below left shoulder
            pose_3d[3] = [pose_3d[1, 0], pose_3d[1, 1] + 0.95, 0.0]
            pose_3d[5] = [pose_3d[1, 0], pose_3d[1, 1] + 1.85, 0.0]

        return pose_3d, lh_3d, rh_3d, active_side

    elif dim_mode == 'one_hand_3d':
        pts = frame_data.reshape(29, 3)
        pose_3d = pts[0:8]
        hand_3d = pts[8:29]

        if dominant_side is not None:
            active_side = dominant_side
        else:
            hand_wrist = hand_3d[0, :2]
            l_wrist = pose_3d[5, :2]
            r_wrist = pose_3d[6, :2]
            d_l = np.linalg.norm(hand_wrist - l_wrist)
            d_r = np.linalg.norm(hand_wrist - r_wrist)
            active_side = 'leftHand' if d_l <= d_r else 'rightHand'

        if active_side == 'leftHand':
            lh_3d = hand_3d
            rh_3d = None
            # Inactive right arm hangs straight down below right shoulder
            pose_3d[4] = [pose_3d[2, 0], pose_3d[2, 1] + 0.95, 0.0]
            pose_3d[6] = [pose_3d[2, 0], pose_3d[2, 1] + 1.85, 0.0]
        else:
            lh_3d = None
            rh_3d = hand_3d
            # Inactive left arm hangs straight down below left shoulder
            pose_3d[3] = [pose_3d[1, 0], pose_3d[1, 1] + 0.95, 0.0]
            pose_3d[5] = [pose_3d[1, 0], pose_3d[1, 1] + 1.85, 0.0]

        return pose_3d, lh_3d, rh_3d, active_side

    elif dim_mode == 'two_hands_2d':
        pts = frame_data.reshape(50, 2)
        pose_2d = pts[0:8]
        lh_2d = pts[8:29]
        rh_2d = pts[29:50]

        pose_3d = synthesize_3d_depth_pose(pose_2d)

        lh_active = np.linalg.norm(lh_2d) > 1e-4
        rh_active = np.linalg.norm(rh_2d) > 1e-4

        lh_3d = synthesize_3d_depth_hand(lh_2d, wrist_z_base=pose_3d[5, 2]) if lh_active else None
        rh_3d = synthesize_3d_depth_hand(rh_2d, wrist_z_base=pose_3d[6, 2]) if rh_active else None

        if lh_active and rh_active:
            active_side = 'both'
        elif lh_active:
            active_side = 'leftHand'
        elif rh_active:
            active_side = 'rightHand'
        else:
            active_side = 'none'

        return pose_3d, lh_3d, rh_3d, active_side

    elif dim_mode == 'two_hands_3d':
        pts = frame_data.reshape(50, 3)
        pose_3d = pts[0:8]
        lh_3d_raw = pts[8:29]
        rh_3d_raw = pts[29:50]

        lh_active = np.linalg.norm(lh_3d_raw) > 1e-4
        rh_active = np.linalg.norm(rh_3d_raw) > 1e-4

        lh_3d = lh_3d_raw if lh_active else None
        rh_3d = rh_3d_raw if rh_active else None

        if lh_active and rh_active:
            active_side = 'both'
        elif lh_active:
            active_side = 'leftHand'
        elif rh_active:
            active_side = 'rightHand'
        else:
            active_side = 'none'

        return pose_3d, lh_3d, rh_3d, active_side

    elif dim_mode == 'mediapipe_1662':
        # Holistic extraction if encountered
        # 1662 features: 468*3 face + 33*3 pose + 21*3 lh + 21*3 rh
        # pose starts at 468*3 = 1404
        pose_raw = frame_data[1404:1404 + 99].reshape(33, 3)
        lh_raw = frame_data[1503:1503 + 63].reshape(21, 3)
        rh_raw = frame_data[1566:1566 + 63].reshape(21, 3)

        pose_indices = [0, 11, 12, 13, 14, 15, 16]
        pose_3d = np.zeros((8, 3), dtype=np.float32)
        for i, idx in enumerate(pose_indices):
            pose_3d[i] = pose_raw[idx]
        # Mid-shoulder = (L_Shoulder + R_Shoulder) / 2
        mid = (pose_raw[11] + pose_raw[12]) / 2.0
        pose_3d[7] = mid
        shoulder_width = max(np.linalg.norm(pose_raw[11] - pose_raw[12]), 1e-4)

        # Normalize relative to mid shoulder
        for i in range(8):
            pose_3d[i] = (pose_3d[i] - mid) / shoulder_width

        lh_active = np.linalg.norm(lh_raw) > 1e-4
        rh_active = np.linalg.norm(rh_raw) > 1e-4

        lh_3d = (lh_raw - mid) / shoulder_width if lh_active else None
        rh_3d = (rh_raw - mid) / shoulder_width if rh_active else None

        if lh_active and rh_active:
            active_side = 'both'
        elif lh_active:
            active_side = 'leftHand'
        elif rh_active:
            active_side = 'rightHand'
        else:
            active_side = 'none'

        return pose_3d, lh_3d, rh_3d, active_side

    else:
        raise ValueError(f"Unsupported dimension mode: {dim_mode}")

def _smooth_1d_channel(y):
    """
    Smooth a 1D trajectory using Savitzky-Golay or 2-pass Gaussian filter.
    """
    n = len(y)
    if n < 3:
        return y

    if HAS_SCIPY:
        w = 7 if n >= 7 else (n if n % 2 == 1 else n - 1)
        if w >= 3:
            poly = min(2, w - 1)
            try:
                return savgol_filter(y, window_length=w, polyorder=poly, mode='interp')
            except Exception:
                pass

    # 2-pass Gaussian / Moving average filter (window=5)
    kernel = np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136], dtype=np.float32)
    kernel = kernel / np.sum(kernel)
    pad_w = len(kernel) // 2

    # Pass 1
    pad_y1 = np.pad(y, pad_w, mode='edge')
    pass1 = np.convolve(pad_y1, kernel, mode='valid')

    # Pass 2
    pad_y2 = np.pad(pass1, pad_w, mode='edge')
    pass2 = np.convolve(pad_y2, kernel, mode='valid')
    return pass2

def smooth_landmark_sequence(raw_data, dim_mode):
    """
    Apply Savitzky-Golay / 2-pass Gaussian temporal smoothing across the time axis
    of each .npy sequence (shape: frames, D), while cleanly preserving missing/zero hand masks.
    """
    T, D = raw_data.shape
    if T < 3:
        return raw_data

    # Map D into (num_points, C)
    if dim_mode == 'one_hand_2d':
        num_points, C = 29, 2
    elif dim_mode == 'one_hand_3d':
        num_points, C = 29, 3
    elif dim_mode == 'two_hands_2d':
        num_points, C = 50, 2
    elif dim_mode == 'two_hands_3d':
        num_points, C = 50, 3
    elif dim_mode == 'mediapipe_1662':
        out = np.copy(raw_data)
        for col in range(D):
            col_data = out[:, col]
            if np.linalg.norm(col_data) > 1e-4:
                out[:, col] = _smooth_1d_channel(col_data)
        return out
    else:
        return raw_data

    pts = raw_data.reshape(T, num_points, C).copy()

    # Pre-pass 1: Bridge 1-frame and 2-frame zero dropouts for hand joints
    hand_start = 8
    for j in range(hand_start, num_points):
        mags = np.linalg.norm(pts[:, j, :], axis=1)
        for t in range(1, T - 1):
            if mags[t] <= 1e-4 and mags[t - 1] > 1e-4:
                if mags[t + 1] > 1e-4:
                    # 1-frame gap: linear interpolation
                    pts[t, j, :] = 0.5 * (pts[t - 1, j, :] + pts[t + 1, j, :])
                    mags[t] = np.linalg.norm(pts[t, j, :])
                elif t + 2 < T and mags[t + 2] > 1e-4:
                    # 2-frame gap: linear interpolation
                    pts[t, j, :] = (2.0 / 3.0) * pts[t - 1, j, :] + (1.0 / 3.0) * pts[t + 2, j, :]
                    pts[t + 1, j, :] = (1.0 / 3.0) * pts[t - 1, j, :] + (2.0 / 3.0) * pts[t + 2, j, :]
                    mags[t] = np.linalg.norm(pts[t, j, :])
                    mags[t + 1] = np.linalg.norm(pts[t + 1, j, :])

    # Pre-pass 2: In two_hands mode, detect and correct cross-body hand label swapping
    if 'two_hands' in dim_mode and num_points == 50:
        for t in range(T):
            lh_mag = np.linalg.norm(pts[t, 8:29, :])
            rh_mag = np.linalg.norm(pts[t, 29:50, :])
            if lh_mag > 1e-4 and rh_mag > 1e-4:
                l_wrist = pts[t, 5, :2]
                r_wrist = pts[t, 6, :2]
                lh_wrist = pts[t, 8, :2]
                rh_wrist = pts[t, 29, :2]
                d_normal = np.linalg.norm(lh_wrist - l_wrist) + np.linalg.norm(rh_wrist - r_wrist)
                d_swapped = np.linalg.norm(lh_wrist - r_wrist) + np.linalg.norm(rh_wrist - l_wrist)
                if d_swapped + 0.35 < d_normal:
                    # MediaPipe swapped left and right hands! Swap them back:
                    tmp = pts[t, 8:29, :].copy()
                    pts[t, 8:29, :] = pts[t, 29:50, :]
                    pts[t, 29:50, :] = tmp

    smoothed_pts = np.copy(pts)

    # Point by point temporal smoothing across time
    for j in range(num_points):
        # Determine active frames for point j
        magnitudes = np.linalg.norm(pts[:, j, :], axis=1)
        active_mask = magnitudes > 1e-4

        if not np.any(active_mask):
            smoothed_pts[:, j, :] = 0.0
            continue

        # Find contiguous active segments [start_idx, end_idx)
        diffs = np.diff(active_mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        if active_mask[0]:
            starts = np.insert(starts, 0, 0)
        ends = np.where(diffs == -1)[0] + 1
        if active_mask[-1]:
            ends = np.append(ends, T)

        for s_idx, e_idx in zip(starts, ends):
            seg_len = e_idx - s_idx
            if seg_len < 3:
                continue

            for c in range(C):
                channel_seg = pts[s_idx:e_idx, j, c]
                smoothed_pts[s_idx:e_idx, j, c] = _smooth_1d_channel(channel_seg)

        # Enforce exact zeros for inactive frames
        smoothed_pts[~active_mask, j, :] = 0.0

    return smoothed_pts.reshape(T, D)

def convert_motion_file(npy_path, word_name, category):
    """
    Convert a single .npy file to motion JSON dict.
    """
    raw_data = np.load(npy_path, allow_pickle=True)

    # Ensure shape is 2D (frames, D)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    frames_count, D = raw_data.shape

    # Determine dimension mode
    if D == 58:
        dim_mode = 'one_hand_2d'
    elif D == 87:
        dim_mode = 'one_hand_3d'
    elif D == 100:
        dim_mode = 'two_hands_2d'
    elif D == 150:
        dim_mode = 'two_hands_3d'
    elif D >= 1629:
        dim_mode = 'mediapipe_1662'
    else:
        # Heuristic fallback based on category
        if category == 'one_hand':
            dim_mode = 'one_hand_2d' if D % 2 == 0 else 'one_hand_3d'
        else:
            dim_mode = 'two_hands_2d' if D % 2 == 0 else 'two_hands_3d'

    # Replicate static 1-frame motions to 30 frames
    if frames_count == 1:
        raw_data = np.repeat(raw_data, 30, axis=0)
        frames_count = 30

    # Apply Savitzky-Golay / Gaussian temporal smoothing across time axis
    raw_data = smooth_landmark_sequence(raw_data, dim_mode)

    # For one_hand mode: determine sequence-wide dominant hand side to prevent mid-sign hand flipping
    dominant_side = None
    if 'one_hand' in dim_mode:
        pts_check = raw_data.reshape(frames_count, 29, -1)
        total_dl = 0.0
        total_dr = 0.0
        for f_idx in range(frames_count):
            hw = pts_check[f_idx, 8, :2]
            lw = pts_check[f_idx, 5, :2]
            rw = pts_check[f_idx, 6, :2]
            total_dl += float(np.linalg.norm(hw - lw))
            total_dr += float(np.linalg.norm(hw - rw))
        dominant_side = 'leftHand' if total_dl <= total_dr else 'rightHand'

    frames_list = []
    for f_idx in range(frames_count):
        f_vec = raw_data[f_idx]
        pose_3d, lh_3d, rh_3d, active_side = process_single_frame(f_vec, dim_mode, dominant_side=dominant_side)

        frame_dict = {
            "frameIndex": f_idx,
            "pose": [[round(float(c), 5) for c in pt] for pt in pose_3d],
            "activeHandSide": active_side,
            "leftHand": [[round(float(c), 5) for c in pt] for pt in lh_3d] if lh_3d is not None else None,
            "rightHand": [[round(float(c), 5) for c in pt] for pt in rh_3d] if rh_3d is not None else None
        }
        frames_list.append(frame_dict)

    result = {
        "word": word_name,
        "category": category,
        "mode": "one_hand" if "one_hand" in dim_mode else "two_hands",
        "fps": 30,
        "frameCount": frames_count,
        "hasDepth": True,
        "sourceShape": list(raw_data.shape),
        "frames": frames_list
    }
    return result

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_root = os.path.dirname(script_dir)           # char/models
    workspace_root = os.path.abspath(os.path.join(models_root, "..", ".."))

    output_motions_dir = os.path.join(models_root, "public", "motions")
    output_models_dir = os.path.join(models_root, "public", "models")
    os.makedirs(output_motions_dir, exist_ok=True)
    os.makedirs(output_models_dir, exist_ok=True)

    print(f"==================================================")
    print(f"VSL 3D Pipeline Initializing...")
    print(f"Workspace Root (READ-ONLY): {workspace_root}")
    print(f"Target Motions Directory:  {output_motions_dir}")
    print(f"Target Models Directory:   {output_models_dir}")
    print(f"==================================================")

    # Phase 2 Acquisition: ensure AvatarSample_A.vrm
    ensure_vrm_model(output_models_dir, workspace_root)

    # Phase 1: Scan data folders (read-only)
    search_dirs = [
        ("one_hand", os.path.join(workspace_root, "data", "one_hand")),
        ("two_hands", os.path.join(workspace_root, "data", "two_hands")),
        ("data_npy", os.path.join(workspace_root, "data_npy")),
        ("MP_Data", os.path.join(workspace_root, "MP_Data")),
        ("dataset", os.path.join(workspace_root, "dataset")),
        ("actions", os.path.join(workspace_root, "actions"))
    ]

    manifest_entries = []
    processed_words = set()

    for category, base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue

        print(f"\nScanning category [{category}] at {base_dir}...")
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        subdirs.sort(key=natural_sort_key)

        for word in subdirs:
            word_dir = os.path.join(base_dir, word)
            # Find all .npy files in word_dir
            npy_files = []
            for root, dirs, files in os.walk(word_dir):
                for f in files:
                    if f.lower().endswith(".npy"):
                        npy_files.append(os.path.join(root, f))

            if not npy_files:
                continue

            # Pick ONLY the first .npy file in natural order
            npy_files.sort(key=natural_sort_key)
            first_npy = npy_files[0]

            out_filename = f"{word}.json"
            out_filepath = os.path.join(output_motions_dir, out_filename)

            try:
                motion_data = convert_motion_file(first_npy, word, category)
                with open(out_filepath, "w", encoding="utf-8") as f:
                    json.dump(motion_data, f, ensure_ascii=False, indent=2)

                manifest_entry = {
                    "id": re.sub(r'[\s\W]+', '_', word).strip('_'),
                    "word": word,
                    "category": category,
                    "mode": motion_data["mode"],
                    "file": out_filename,
                    "path": f"motions/{out_filename}",
                    "frameCount": motion_data["frameCount"],
                    "fps": motion_data["fps"],
                    "hasDepth": True
                }
                manifest_entries.append(manifest_entry)
                processed_words.add(word)
                print(f"  [OK] Converted '{word}' ({category}) -> {out_filename} ({motion_data['frameCount']} frames)")

            except Exception as e:
                print(f"  [ERROR] Failed to convert '{word}': {e}")

    # Generate manifest.json
    manifest_path = os.path.join(output_motions_dir, "manifest.json")
    manifest_data = {
        "generatedAt": "2026-11-14",
        "totalWords": len(manifest_entries),
        "words": manifest_entries
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    print(f"\n==================================================")
    print(f"Pipeline Complete: {len(manifest_entries)} words converted.")
    print(f"Manifest written to: {manifest_path}")
    print(f"==================================================")

if __name__ == "__main__":
    main()

