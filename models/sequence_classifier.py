#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module chuyển đổi và tối ưu hóa mô hình Keras sang TensorFlow Lite (TFLite).
"""

import os
import tensorflow as tf


def convert_to_tflite(keras_model, output_path, float16_quantization=True):
    """
    Chuyển đổi mô hình Keras sang định dạng .tflite với tối ưu lượng tử hóa Float16.

    :param keras_model: Mô hình tf.keras.Model đã huấn luyện
    :param output_path: Đường dẫn tệp .tflite cần xuất ra
    :param float16_quantization: Bật lượng tử hóa Float16 (giảm 50% kích thước mô hình)
    :return: Nội dung byte của tệp tflite
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if float16_quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[SUCCESS] Đã xuất mô hình TFLite tại: {output_path} (Kích thước: {size_kb:.2f} KB)")
    return tflite_model
