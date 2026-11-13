#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper Class Suy luận Mô hình Nhánh 2 Tay (Two-Hands Classifier).
Sử dụng TFLite Interpreter cho nhận diện các từ cử chỉ 2 bàn tay kết hợp thân trên.
"""

import os
import numpy as np
import tensorflow as tf


class TwoHandsClassifier:
    """
    Trình suy luận TFLite cho Nhánh 2 Tay kết hợp Thân trên (Two-Hands Branch).
    """

    def __init__(self, model_path="model/two_hands_classifier/two_hands_classifier.tflite", num_threads=2):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.is_loaded = False

        if os.path.exists(model_path):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.expected_dim = int(self.input_details[0]['shape'][-1])
                self.is_loaded = True
                print(f"[TWO-HANDS CLASSIFIER] Loaded TFLite model from: {model_path} (Expected dim: {self.expected_dim})")
            except Exception as e:
                print(f"[TWO-HANDS CLASSIFIER ERROR] Failed to load model: {e}")
        else:
            print(f"[TWO-HANDS CLASSIFIER WARNING] Model file not found at '{model_path}'.")

    def predict(self, input_data):
        """
        Dự đoán lớp cử chỉ 2 tay.

        :param input_data: np.ndarray kích thước (num_features,) hoặc (30, num_features)
        :return: (class_id, confidence_score, probabilities)
        """
        if not self.is_loaded:
            return -1, 0.0, np.array([])

        arr = np.array(input_data, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)  # Shape (1, num_features)
        elif arr.ndim == 2 and arr.shape[0] == 30:
            arr = np.expand_dims(arr, axis=0)  # Shape (1, 30, num_features)

        # Kiểm tra khớp số chiều đặc trưng giữa thuật toán và model TFLite hiện có
        if arr.shape[-1] != self.expected_dim:
            if not getattr(self, '_dim_warned', False):
                print(f"[TWO-HANDS CLASSIFIER CANH BAO] So chieu dac trung hien tai ({arr.shape[-1]}D) khac voi model TFLite cu ({self.expected_dim}D). Bo qua suy luan cho den khi ban thu thap data moi va train lai notebooks/train_two_hands.ipynb.")
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
            if not getattr(self, '_dim_warned', False):
                print(f"[TWO-HANDS CLASSIFIER ERROR] Không thể suy luận: {e}")
                self._dim_warned = True
            return -1, 0.0, np.array([])
