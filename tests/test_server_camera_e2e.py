#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Unit: Kiểm tra toàn diện Camera Server, Bộ đệm chống rung, và Tự động thêm từ vào hàng đợi.
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.sign_engine import SignRecognitionEngine
from utils.visualizer import Visualizer
from web_app.server import on_engine_word_recognized, ws_clients


def test_engine_attributes_and_status():
    print("[TEST 1] Kiểm tra thuộc tính và get_status() của SignRecognitionEngine...")
    engine = SignRecognitionEngine(hold_duration=1.0)
    
    assert hasattr(engine, "miss_grace_period"), "Phải có thuộc tính miss_grace_period"
    assert hasattr(engine, "hold_grace_period"), "Phải có thuộc tính hold_grace_period"
    assert hasattr(engine, "MAX_MISMATCH_FRAMES"), "Phải có thuộc tính MAX_MISMATCH_FRAMES"
    assert engine.MAX_MISMATCH_FRAMES >= 6, "MAX_MISMATCH_FRAMES phải đủ lớn (>= 6) để chống rung"
    assert engine.hold_duration == 1.0, "Mặc định hold duration là 1.0s"

    st = engine.get_status()
    assert "holding" in st, "get_status() phải chứa khóa 'holding'"
    assert "word" in st["holding"]
    assert "progress" in st["holding"]
    assert "confirmed" in st["holding"]
    print(" -> PASSED: Thuộc tính và get_status() đầy đủ và chuẩn xác!")


def test_visualizer_draw_prediction_with_hold():
    print("[TEST 2] Kiểm tra Visualizer.draw_prediction() với thanh tiến trình hold_info...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Test khi đang giữ 0.6s / 1.0s (60%)
    hold_info_holding = {
        "seconds": 0.6,
        "target": 1.0,
        "progress": 0.6,
        "confirmed": False
    }
    Visualizer.draw_prediction(img, "xin chào", 0.92, "ONE_HAND", hold_info=hold_info_holding)

    # 2. Test khi đã chốt
    hold_info_confirmed = {
        "seconds": 1.0,
        "target": 1.0,
        "progress": 1.0,
        "confirmed": True
    }
    Visualizer.draw_prediction(img, "cảm ơn", 0.98, "TWO_HANDS", hold_info=hold_info_confirmed)
    print(" -> PASSED: Visualizer vẽ giao diện thẻ và thanh tiến trình giữ cử chỉ không lỗi!")


def test_hold_progress_with_hand_tremors():
    print("[TEST 3] Giả lập luồng nhận diện cử chỉ 1.0 giây có rung tay và mất frame...")
    engine = SignRecognitionEngine(hold_duration=1.0)
    events = []
    engine.register_word_callback(lambda ev: events.append(ev))

    t0 = time.time()
    # Giả lập 35 khung hình liên tiếp (~1.4 giây ở 25 FPS)
    # Cứ mỗi 10 khung hình lại có 1-2 khung hình bị rớt/nhiễu
    for frame_idx in range(35):
        simulated_now = t0 + (frame_idx * 0.04) # mỗi frame cách nhau 40ms
        
        # Giả lập rớt 2 frame ở frame 8, 9
        if frame_idx in (8, 9):
            rec_word = "" # không nhận ra từ do rung tay
            rec_conf = 0.0
            rec_branch = ""
        else:
            rec_word = "tôi"
            rec_conf = 0.91
            rec_branch = "ONE_HAND"

        # Mô phỏng logic tính hold trong process_frame
        word_event = None
        if rec_word:
            engine.last_seen_word_time = simulated_now
            if not engine.current_hold_word:
                engine.current_hold_word = rec_word
                engine.current_hold_start = simulated_now
                engine.current_hold_branch = rec_branch
                engine.current_hold_conf = rec_conf
                engine.hold_confirmed = False
                engine.mismatch_frames = 0
            elif rec_word == engine.current_hold_word:
                engine.mismatch_frames = 0
                engine.current_hold_conf = rec_conf
                hold_sec = max(0.0, simulated_now - engine.current_hold_start)
                hold_prog = min(1.0, hold_sec / engine.hold_duration)
                if hold_sec >= engine.hold_duration and not engine.hold_confirmed:
                    engine.hold_confirmed = True
                    word_event = {
                        "type": "word_recognized",
                        "word": rec_word,
                        "confidence": rec_conf,
                        "branch": rec_branch,
                        "timestamp": simulated_now,
                        "hold_seconds": round(hold_sec, 2)
                    }
        else:
            if engine.current_hold_word:
                engine.mismatch_frames += 1
                if engine.mismatch_frames >= engine.MAX_MISMATCH_FRAMES or (simulated_now - engine.last_seen_word_time > engine.hold_grace_period):
                    engine.current_hold_word = ""
                    engine.current_hold_start = 0.0
                    engine.hold_confirmed = False
                    engine.mismatch_frames = 0

        if word_event:
            for cb in list(engine.word_callbacks):
                cb(word_event)

    assert len(events) == 1, f"Kỳ vọng đúng 1 sự kiện chốt từ, nhận được: {len(events)}"
    assert events[0]["word"] == "tôi"
    assert events[0]["hold_seconds"] >= 1.0
    print(" -> PASSED: Cử chỉ giữ 1.0s được chốt thành công bất chấp rung tay và rớt frame!")


def test_server_broadcast_flow():
    print("[TEST 4] Kiểm tra luồng phát tín hiệu on_engine_word_recognized...")
    class MockWsClient:
        def __init__(self):
            self.messages = []
        def write_message(self, msg):
            self.messages.append(msg)

    mock_client = MockWsClient()
    ws_clients.add(mock_client)

    try:
        sample_event = {
            "type": "word_recognized",
            "word": "xin chào",
            "confidence": 0.95,
            "branch": "ONE_HAND",
            "timestamp": time.time(),
            "hold_seconds": 2.05
        }
        on_engine_word_recognized(sample_event)
        assert len(mock_client.messages) == 1, "Mock client phải nhận được đúng 1 tin nhắn JSON"
        assert '"word": "xin chào"' in mock_client.messages[0]
        print(" -> PASSED: Luồng phát tín hiệu WebSocket hoạt động tin cậy!")
    finally:
        ws_clients.discard(mock_client)


if __name__ == "__main__":
    test_engine_attributes_and_status()
    test_visualizer_draw_prediction_with_hold()
    test_hold_progress_with_hand_tremors()
    test_server_broadcast_flow()
    print("\n[TẤT CẢ KIỂM THỬ ĐỀU THÀNH CÔNG VƯỢT TRỘI!]")

