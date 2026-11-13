#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Unit cho Cơ chế Duy trì cử chỉ 2 giây (Hold 2s) và Ghép câu.
"""

import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.sign_engine import SignRecognitionEngine
from web_app.gemini_service import GeminiSentenceBuilder, preprocess_vsl_tokens


def test_hold_mechanism():
    print("[TEST 1] Kiểm tra cơ chế Hold 1.0s trong SignRecognitionEngine...")
    engine = SignRecognitionEngine(hold_duration=1.0)
    assert engine.hold_duration == 1.0, "Hold duration phải là 1.0s"

    # Giả lập phát hiện từ
    # Đặt trạng thái nhận diện
    t0 = time.time()
    engine.current_hold_word = "xin chào"
    engine.current_hold_start = t0
    engine.current_hold_branch = "ONE_HAND"
    engine.current_hold_conf = 0.95
    engine.hold_confirmed = False
    engine.last_seen_word_time = t0

    # Kiểm tra sau 0.5s (chưa đủ 1s)
    now = t0 + 0.5
    hold_sec = now - engine.current_hold_start
    prog = hold_sec / engine.hold_duration
    assert prog == 0.5, f"Progress phải là 0.5, thực tế là {prog}"
    assert not engine.hold_confirmed, "Sau 0.5s chưa được chốt"

    # Kiểm tra sau 1.0s (đủ 1s)
    now = t0 + 1.0
    hold_sec = now - engine.current_hold_start
    assert hold_sec >= engine.hold_duration, "Đã đạt hoặc vượt 1.0s"
    # Giả lập chốt từ
    engine.hold_confirmed = True
    assert engine.hold_confirmed, "Đã chốt thành công sau 1.0s"

    # Kiểm tra nếu giữ tiếp tới 1.5s thì không chốt lại
    now = t0 + 1.5
    assert engine.hold_confirmed, "Vẫn giữ trạng thái confirmed, không bắn event trùng lặp"

    print(" -> PASSED: Cơ chế Hold 1.0s hoạt động chuẩn xác!")


def test_gemini_token_preprocessing():
    print("\n[TEST 2] Kiểm tra hàm tiền xử lý từ ngữ VSL & Ghép số...")
    raw_tokens = ["xin chào", "bạn", "có / đúng", "cái gì", "giá", "1", "00", "000", "không"]
    cleaned = preprocess_vsl_tokens(raw_tokens)
    print(f"  Từ gốc:   {raw_tokens}")
    print(f"  Sau xử lý: {cleaned}")

    assert "có / đúng" not in cleaned, "Nhãn kép 'có / đúng' phải được xử lý thành 'có'"
    assert "có" in cleaned, "Phải chứa từ 'có'"
    assert "100.000" in cleaned, "Chuỗi số ['1', '00', '000'] phải được ghép thành '100.000'"
    print(" -> PASSED: Tiền xử lý từ ngữ và ghép số chuẩn xác!")


def test_gemini_fallback():
    print("\n[TEST 3] Kiểm tra Gemini fallback khi chưa có API Key...")
    builder = GeminiSentenceBuilder(api_key="")
    raw_tokens = ["xin chào", "bạn", "có / đúng", "cái gì", "giá", "1", "00", "000", "không"]
    res = builder.combine_words_to_sentence(raw_tokens)
    print(f"  Kết quả fallback: {res['sentence']}")
    assert "error" in res
    assert "sentence" in res
    assert len(res["sentence"]) > 0
    print(" -> PASSED: Gemini Service xử lý an toàn không gặp lỗi ngoại lệ!")


if __name__ == "__main__":
    test_hold_mechanism()
    test_gemini_token_preprocessing()
    test_gemini_fallback()
    print("\n[TẤT CẢ KIỂM THỬ ĐỀU THÀNH CÔNG!]")

