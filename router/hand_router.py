#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Bộ định tuyến bàn tay (Hand Router).
Xác định chế độ làm việc: 1 tay (ONE_HAND), 2 tay (TWO_HANDS) hoặc KHÔNG CÓ TAY (NO_HAND).
Sử dụng bộ đệm làm mịn (Buffer Voting) để tránh dao động phân nhánh khi 1 bàn tay tạm chớp nháy.
"""

from collections import deque


class HandRouter:
    """
    Bộ định tuyến cử chỉ tay thông minh (Hierarchical Router).
    """

    MODE_NO_HAND = "NO_HAND"
    MODE_ONE_HAND = "ONE_HAND"
    MODE_TWO_HANDS = "TWO_HANDS"

    def __init__(self, history_len=5):
        """
        :param history_len: Độ dài hàng đợi làm mịn để quyết định nhánh ổn định.
        """
        self.history = deque(maxlen=history_len)

    def route(self, results):
        """
        Phân tích kết quả từ MediaPipe Holistic hoặc Hands để xác định nhánh xử lý.

        :param results: Kết quả MediaPipe (Holistic hoặc Hands)
        :return: Chuỗi chế độ: "ONE_HAND", "TWO_HANDS", hoặc "NO_HAND"
        """
        num_hands = 0

        # Nếu dùng MediaPipe Holistic
        if hasattr(results, 'left_hand_landmarks') and hasattr(results, 'right_hand_landmarks'):
            if results.left_hand_landmarks is not None:
                num_hands += 1
            if results.right_hand_landmarks is not None:
                num_hands += 1

        # Nếu dùng MediaPipe Hands
        elif hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

        # Xác định chế độ khung hình hiện tại
        if num_hands >= 2:
            current_mode = self.MODE_TWO_HANDS
        elif num_hands == 1:
            current_mode = self.MODE_ONE_HAND
        else:
            current_mode = self.MODE_NO_HAND

        self.history.append(current_mode)

        # Bầu chọn chế độ chiếm đa số trong bộ đệm làm mịn (Majority Voting)
        count_two = self.history.count(self.MODE_TWO_HANDS)
        count_one = self.history.count(self.MODE_ONE_HAND)

        if count_two > 0 and count_two >= count_one:
            return self.MODE_TWO_HANDS
        elif count_one > 0:
            return self.MODE_ONE_HAND
        else:
            return self.MODE_NO_HAND
