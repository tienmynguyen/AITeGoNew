#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trình suy luận Mô hình TFLite cho Nhánh 1 Tay & 2 Tay (Classifiers).
Tự động hỗ trợ cả TensorFlow Lite Interpreter và tflite-runtime.
"""

import os
import numpy as np

# Thử nạp TFLite Interpreter từ tensorflow hoặc tflite_runtime
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        Interpreter = None


class BaseTFLiteClassifier:
    """Lớp cơ sở cho trình suy luận TFLite."""
    def __init__(self, model_path: str, branch_name: str, num_threads: int = 2):
        self.model_path = model_path
        self.branch_name = branch_name
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.expected_dim = 0
        self.is_loaded = False
        self._dim_warned = False

        if not os.path.exists(model_path):
            print(f"[{branch_name.upper()} WARNING] Không tìm thấy file model: '{model_path}'")
            return

        if Interpreter is None:
            print(f"[{branch_name.upper()} ERROR] Chưa cài đặt tensorflow hoặc tflite-runtime!")
            return

        try:
            self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.expected_dim = int(self.input_details[0]['shape'][-1])
            self.is_loaded = True
            print(f"[{branch_name.upper()}] Đã nạp model TFLite: {os.path.basename(model_path)} (Đặc trưng: {self.expected_dim}D)")
        except Exception as e:
            print(f"[{branch_name.upper()} ERROR] Không thể khởi tạo interpreter: {e}")

    def predict(self, input_data):
        """
        Dự đoán lớp cử chỉ.
        :param input_data: list hoặc np.ndarray (1D đặc trưng hoặc 2D sequence 30 frames)
        :return: (class_id, confidence, probabilities)
        """
        if not self.is_loaded:
            return -1, 0.0, np.array([])

        arr = np.array(input_data, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 2 and arr.shape[0] == 30:
            arr = np.expand_dims(arr, axis=0)

        if arr.shape[-1] != self.expected_dim:
            if not self._dim_warned:
                print(f"[{self.branch_name.upper()} CẢNH BÁO] Số chiều đặc trưng ({arr.shape[-1]}D) không khớp với model ({self.expected_dim}D).")
                self._dim_warned = True
            return -1, 0.0, np.array([])

        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], arr)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            probs = output_data[0]
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            return class_id, confidence, probs
        except Exception as e:
            if not self._dim_warned:
                print(f"[{self.branch_name.upper()} ERROR] Lỗi trong quá trình suy luận: {e}")
                self._dim_warned = True
            return -1, 0.0, np.array([])


class OneHandClassifier(BaseTFLiteClassifier):
    """Trình suy luận cho Nhánh 1 Tay (One-Hand Branch, 58D)."""
    def __init__(self, model_path: str, num_threads: int = 2):
        super().__init__(model_path=model_path, branch_name="ONE-HAND CLASSIFIER", num_threads=num_threads)


class TwoHandsClassifier(BaseTFLiteClassifier):
    """Trình suy luận cho Nhánh 2 Tay (Two-Hands Branch, 100D)."""
    def __init__(self, model_path: str, num_threads: int = 2):
        super().__init__(model_path=model_path, branch_name="TWO-HANDS CLASSIFIER", num_threads=num_threads)

