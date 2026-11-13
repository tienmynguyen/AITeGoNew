#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import time
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.sign_engine import SignRecognitionEngine

def run_test():
    engine = SignRecognitionEngine()
    print("SignRecognitionEngine initialized successfully.")
    print(f"One-hand labels count: {len(engine.one_hand_labels)}")
    print(f"Two-hands labels count: {len(engine.two_hands_labels)}")

    test_digits = ["1", "2", "3", "5", "6", "7", "8"]
    for d in test_digits:
        files = glob.glob(f"data/two_hands/{d}/*.npy")
        if not files:
            continue
        raw_data = np.load(files[0])  # (30, 100)

        # 1. Test với chuỗi trọn vẹn 30 frame
        t0 = time.time()
        cid, conf, _ = engine.two_hands_clf.predict(raw_data)
        t_full = (time.time() - t0) * 1000
        word_full = engine.two_hands_labels[cid]

        # 2. Test dự đoán sớm siêu nhanh với chỉ 10 frame (Sliding Window Padding)
        sub_seq = list(raw_data[:10])
        padded = engine._pad_sequence(sub_seq, 30)
        t1 = time.time()
        cid_early, conf_early, _ = engine.two_hands_clf.predict(padded)
        t_early = (time.time() - t1) * 1000
        word_early = engine.two_hands_labels[cid_early]

        print(f"Digit [{d}]:")
        print(f"  - Full 30 frames:  Pred = '{word_full}' (conf = {conf:.2f}, time = {t_full:.1f}ms)")
        print(f"  - Early 10 frames: Pred = '{word_early}' (conf = {conf_early:.2f}, time = {t_early:.1f}ms)")

    print("\n[TEST COMPLETED SUCCESSFULLY]")

if __name__ == "__main__":
    run_test()

