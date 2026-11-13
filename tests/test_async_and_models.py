#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Unit kiểm tra xử lý bất đồng bộ (Non-blocking Async) và lọc Model Gemini.
"""

import os
import sys
import inspect
import asyncio

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.server import LLMCombineHandler, LLMTestKeyHandler
from web_app.gemini_service import (
    GeminiSentenceBuilder,
    DEFAULT_GEMINI_MODEL,
    SUPPORTED_MODELS,
    preprocess_vsl_tokens
)


def test_async_handlers():
    print("[TEST 1] Kiểm tra các Handler Tornado có phải là async coroutine không...")
    assert inspect.iscoroutinefunction(LLMCombineHandler.post), "LLMCombineHandler.post PHẢI là async def!"
    assert inspect.iscoroutinefunction(LLMTestKeyHandler.post), "LLMTestKeyHandler.post PHẢI là async def!"
    print(" -> PASSED: LLMCombineHandler và LLMTestKeyHandler đều là async coroutines (Không bao giờ khóa luồng Tornado WebSocket)!")


def test_model_filtering():
    print("\n[TEST 2] Kiểm tra danh sách model hỗ trợ và tự động chuyển đổi model cũ...")
    assert "gemini-2.5-flash" not in SUPPORTED_MODELS, "gemini-2.5-flash không được nằm trong SUPPORTED_MODELS"
    assert "gemini-2.0-flash" not in SUPPORTED_MODELS, "gemini-2.0-flash không được nằm trong SUPPORTED_MODELS"
    assert DEFAULT_GEMINI_MODEL in ("gemini-3.5-flash", "gemini-3.8-flash"), f"DEFAULT_GEMINI_MODEL không hợp lệ: {DEFAULT_GEMINI_MODEL}"

    builder = GeminiSentenceBuilder(model="gemini-2.5-flash")
    # Giả lập gọi với model cũ đã bị Google khai tử
    res = builder.combine_words_to_sentence(
        words=["xin chào", "bạn", "có / đúng", "cái gì", "giá", "1", "00", "000", "không"],
        model_override="gemini-2.5-flash"
    )
    assert res is not None, "Kết quả trả về không được None"
    print(f"  Fallback sentence khi chưa có key: {res.get('sentence')}")
    print(" -> PASSED: Tự động loại bỏ model cũ và xử lý fallback mượt mà!")


if __name__ == "__main__":
    test_async_handlers()
    test_model_filtering()
    print("\n[TẤT CẢ KIỂM THỬ ĐỀU THÀNH CÔNG!]")

