#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque
import cv2 as cv


class CvFpsCalc:
    """Bộ đếm tốc độ khung hình (FPS) trung bình trượt."""
    def __init__(self, buffer_len: int = 10):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self) -> float:
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(different_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        return round(fps, 1)

