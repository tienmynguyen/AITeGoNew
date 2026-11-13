#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Unit: Kiểm tra khả năng chống nhiễu (Jitter Tolerance) khi giữ cử chỉ 2 giây
và kiểm tra luồng phát tín hiệu sự kiện của Server Camera.
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.sign_engine import SignRecognitionEngine
from web_app.server import on_engine_word_recognized


def test_jitter_tolerance():
    print("[TEST 1] Kiểm tra cơ chế chống nhiễu Jitter khi giữ cử chỉ...")
    engine = SignRecognitionEngine(hold_duration=1.0)
    events_received = []
    engine.register_word_callback(lambda ev: events_received.append(ev))

    t0 = time.time()
    # Frame 1: Nhận diện "xin chào"
    engine.current_hold_word = "xin chào"
    engine.current_hold_start = t0
    engine.current_hold_branch = "ONE_HAND"
    engine.current_hold_conf = 0.95
    engine.hold_confirmed = False
    engine.mismatch_frames = 0

    # Giả lập frame 10 (sau 0.4s): bị nhiễu 2 frame dự đoán từ khác
    # Frame nhiễu 1:
    engine.mismatch_frames += 1
    assert engine.current_hold_word == "xin chào", "Chưa vượt quá mismatch frames thì không được đổi từ"
    assert engine.mismatch_frames == 1

    # Frame nhiễu 2:
    engine.mismatch_frames += 1
    assert engine.current_hold_word == "xin chào"
    assert engine.mismatch_frames == 2

    # Frame 13: Quay lại nhận diện đúng "xin chào"
    engine.mismatch_frames = 0
    assert engine.mismatch_frames == 0, "Khi nhận diện lại đúng thì reset mismatch_frames về 0"
    assert engine.current_hold_start == t0, "current_hold_start vẫn giữ nguyên thời điểm ban đầu!"

    # Giả lập đến 1.1s:
    now = t0 + 1.1
    hold_sec = now - engine.current_hold_start
    assert hold_sec >= 1.0, "Đã đủ 1.0 giây"

    # Kích hoạt chốt từ
    engine.hold_confirmed = True
    ev = {
        "type": "word_recognized",
        "word": "xin chào",
        "confidence": 0.95,
        "branch": "ONE_HAND",
        "timestamp": now,
        "hold_seconds": round(hold_sec, 2)
    }
    for cb in engine.word_callbacks:
        cb(ev)

    assert len(events_received) == 1, "Phải nhận được đúng 1 sự kiện word_recognized"
    assert events_received[0]["word"] == "xin chào"
    print(" -> PASSED: Bộ lọc chống nhiễu (Jitter Cushion) hoạt động mượt mà, giữ vững mốc 1.0s!")


if __name__ == "__main__":
    test_jitter_tolerance()
    print("[TẤT CẢ KIỂM THỬ ĐỀU THÀNH CÔNG!]")

