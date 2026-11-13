#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Chuẩn hóa Tọa độ Đặc trưng cho Cả 1 Tay (One-Hand) và 2 Tay (Two-Hands).
Tối ưu hóa bất biến kích thước và gốc tọa độ tương đối.
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import numpy as np


class LandmarkExtractor:
    """
    Bộ trích xuất & chuẩn hóa tọa độ đặc trưng phân nhánh (Hierarchical Landmark Extractor).
    """

    # Chỉ số các điểm Pose thân trên trong MediaPipe Pose (33 điểm):
    # 0: Mũi, 11: Vai trái, 12: Vai phải, 13: Khuỷu trái, 14: Khuỷu phải, 15: Cổ tay trái, 16: Cổ tay phải
    POSE_LANDMARK_INDICES = [0, 11, 12, 13, 14, 15, 16]

    def __init__(self, use_3d=False):
        self.use_3d = use_3d
        self.num_coords = 3 if use_3d else 2

    def _extract_pose(self, results):
        """
        Trích xuất 8 điểm Pose thân trên và tính gốc Mid-Shoulder cùng khoảng cách hai vai.
        """
        pose_pts = []
        mid_shoulder = np.zeros(self.num_coords, dtype=np.float32)
        shoulder_width = 1.0

        if results and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z] if self.use_3d 
                                     else [landmarks[11].x, landmarks[11].y], dtype=np.float32)
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z] if self.use_3d 
                                      else [landmarks[12].x, landmarks[12].y], dtype=np.float32)

            mid_shoulder = (left_shoulder + right_shoulder) / 2.0
            dist = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
            if dist > 1e-6:
                shoulder_width = dist

            for idx in self.POSE_LANDMARK_INDICES:
                lm = landmarks[idx]
                pt = np.array([lm.x, lm.y, lm.z] if self.use_3d else [lm.x, lm.y], dtype=np.float32)
                pose_pts.append(pt)

            pose_pts.append(mid_shoulder)
        else:
            for _ in range(8):
                pose_pts.append(np.zeros(self.num_coords, dtype=np.float32))

        return pose_pts, mid_shoulder, shoulder_width

    def normalize_one_hand(self, results, hand_landmarks=None):
        """
        Chuẩn hóa 29 điểm mốc (8 điểm Thân trên/Pose + 21 điểm Bàn tay hoạt động) cho Nhánh 1 Tay:
          - Gốc tọa độ tương đối (0, 0): Trung điểm hai vai (Mid-Shoulder).
          - Tỉ lệ khoảng cách: Chia cho Khoảng cách hai vai (Shoulder Width).
          - Bảo toàn đầy đủ hình dạng ngón tay đồng thời nắm bắt chính xác
            vị trí không gian của bàn tay so với khuôn mặt, cằm, ngực và thân người.

        :param results: Kết quả MediaPipe Holistic (hoặc NormalizedLandmarkList bàn tay)
        :param hand_landmarks: (Tùy chọn) Bàn tay chỉ định nếu muốn override
        :return: np.ndarray 1D kích thước (58,) cho 2D hoặc (87,) cho 3D.
        """
        # Hỗ trợ tương thích ngược nếu người dùng truyền trực tiếp hand_landmarks
        if hand_landmarks is None and hasattr(results, 'landmark') and not hasattr(results, 'pose_landmarks'):
            hand_landmarks = results
            results = None

        pose_pts, mid_shoulder, shoulder_width = self._extract_pose(results)

        # Xác định bàn tay hoạt động (Active Hand)
        active_hand = hand_landmarks
        if active_hand is None and results is not None:
            lh = getattr(results, 'left_hand_landmarks', None)
            rh = getattr(results, 'right_hand_landmarks', None)
            if lh and rh:
                # Nếu phát hiện cả 2 tay, ưu tiên tay giơ cao hơn (tọa độ y nhỏ hơn)
                active_hand = rh if rh.landmark[0].y <= lh.landmark[0].y else lh
            else:
                active_hand = rh or lh

        hand_pts = []
        if active_hand and hasattr(active_hand, 'landmark') and active_hand.landmark:
            for lm in active_hand.landmark:
                pt = np.array([lm.x, lm.y, lm.z] if self.use_3d else [lm.x, lm.y], dtype=np.float32)
                hand_pts.append(pt)
        else:
            for _ in range(21):
                hand_pts.append(np.zeros(self.num_coords, dtype=np.float32))

        # Chuẩn hóa tất cả 29 điểm theo Mid-Shoulder và Shoulder Width
        normalized_vector = []
        for group in [pose_pts, hand_pts]:
            for pt in group:
                if np.all(pt == 0):
                    normalized_vector.append(pt)
                else:
                    norm_pt = (pt - mid_shoulder) / shoulder_width
                    normalized_vector.append(norm_pt)

        return np.array(normalized_vector, dtype=np.float32).flatten()

    def normalize_two_hands(self, results):
        """
        Chuẩn hóa 50 điểm mốc 2 Bàn tay + Thân trên (MediaPipe Holistic):
          - Gốc tọa độ tương đối (0, 0): Trung điểm hai vai (Mid-Shoulder).
          - Tỉ lệ khoảng cách: Chia cho Khoảng cách hai vai (Shoulder Width).
          - Zero-padding cho bàn tay bị che khuất.

        :param results: Kết quả MediaPipe Holistic
        :return: np.ndarray 1D kích thước (100,) cho 2D hoặc (150,) cho 3D.
        """
        pose_pts, mid_shoulder, shoulder_width = self._extract_pose(results)

        # Tay trái (21 điểm)
        lh_pts = []
        if results and results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                pt = np.array([lm.x, lm.y, lm.z] if self.use_3d else [lm.x, lm.y], dtype=np.float32)
                lh_pts.append(pt)
        else:
            for _ in range(21):
                lh_pts.append(np.zeros(self.num_coords, dtype=np.float32))

        # Tay phải (21 điểm)
        rh_pts = []
        if results and results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                pt = np.array([lm.x, lm.y, lm.z] if self.use_3d else [lm.x, lm.y], dtype=np.float32)
                rh_pts.append(pt)
        else:
            for _ in range(21):
                rh_pts.append(np.zeros(self.num_coords, dtype=np.float32))

        # Chuẩn hóa tất cả 50 điểm mốc theo Mid-Shoulder và Shoulder Width
        normalized_vector = []
        for group in [pose_pts, lh_pts, rh_pts]:
            for pt in group:
                if np.all(pt == 0):
                    normalized_vector.append(pt)
                else:
                    norm_pt = (pt - mid_shoulder) / shoulder_width
                    normalized_vector.append(norm_pt)

        return np.array(normalized_vector, dtype=np.float32).flatten()
