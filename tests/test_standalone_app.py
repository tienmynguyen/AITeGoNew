#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Unit: Kiểm tra toàn diện ứng dụng độc lập standalone_app.
"""

import os
import sys
import time
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.join(ROOT_DIR, "standalone_app")
sys.path.insert(0, APP_DIR)

from core.sign_engine import SignRecognitionEngine
from core.gemini_service import GeminiSentenceBuilder
from core.classifiers import OneHandClassifier, TwoHandsClassifier
from core.hand_router import HandRouter


def test_core_modules():
    print("[TEST 1] Kiểm tra nạp các module độc lập trong standalone_app/core/...")
    router = HandRouter()
    assert router.route(None) == "NO_HAND"

    engine = SignRecognitionEngine(hold_duration=1.0)
    assert engine.hold_duration == 1.0, "Hold duration mặc định là 1.0s"
    assert engine.one_hand_clf.is_loaded, "Model 1 tay phải nạp thành công"
    assert engine.two_hands_clf.is_loaded, "Model 2 tay phải nạp thành công"
    assert len(engine.one_hand_labels) == 10, "Phải có đúng 10 nhãn 1 tay"
    assert len(engine.two_hands_labels) == 28, "Phải có đúng 28 nhãn 2 tay"

    gemini = GeminiSentenceBuilder()
    res = gemini.combine_words_to_sentence(["xin chào", "bạn"])
    assert "sentence" in res and len(res["sentence"]) > 0, "Phải có trường sentence hoàn chỉnh"
    print(f"  Sentence output (fallback): {res['sentence']}")
    print(" -> PASSED: Toàn bộ core modules hoạt động độc lập hoàn hảo!")


def test_standalone_app_file_structure():
    print("[TEST 2] Kiểm tra cấu trúc thư mục standalone_app...")
    expected_files = [
        "models/one_hand_classifier.tflite",
        "models/two_hands_classifier.tflite",
        "models/labels_one_hand.csv",
        "models/labels_two_hands.csv",
        "core/__init__.py",
        "core/classifiers.py",
        "core/hand_router.py",
        "core/normalization.py",
        "core/visualizer.py",
        "core/cvfpscalc.py",
        "core/gemini_service.py",
        "core/sign_engine.py",
        "static/index.html",
        "static/css/style.css",
        "static/js/app.js",
        "app.py",
        "run_app.bat",
        "requirements.txt",
        "README.md"
    ]
    for rel_path in expected_files:
        full_p = os.path.join(APP_DIR, rel_path)
        assert os.path.exists(full_p), f"Thiếu file: {rel_path}"
    print(" -> PASSED: Đầy đủ 100% tất cả các file cấu trúc độc lập!")


if __name__ == "__main__":
    test_core_modules()
    test_standalone_app_file_structure()
    print("\n[TẤT CẢ KIỂM THỬ STANDALONE APP ĐỀU THÀNH CÔNG VƯỢT TRỘI!]")
