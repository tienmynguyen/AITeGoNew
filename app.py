#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ứng dụng Nhận diện Ngôn ngữ Ký hiệu Kiến trúc Phân nhánh (Hierarchical Router Architecture)
Bao gồm:
  - Bộ định tuyến bàn tay (HandRouter): Tự động chuyển đổi giữa Nhánh 1 Tay (One-Hand) và 2 Tay (Two-Hands).
  - Nhánh 1 Tay (OneHandClassifier): Xử lý 42 đặc trưng (Tọa độ tương đối theo Cổ tay).
  - Nhánh 2 Tay (TwoHandsClassifier): Xử lý 100 đặc trưng (MediaPipe Holistic Pose + 2 Hands theo Mid-Shoulder).
  - Trực quan hóa Giao diện (Visualizer) & Bộ thu thập dữ liệu chuỗi 30 khung hình.
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import csv
import time
import argparse
from collections import deque

import cv2 as cv
import numpy as np

# Import trực tiếp MediaPipe Solutions
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

# Import các Module Chuyên biệt theo Cấu trúc Phân nhánh
from router.hand_router import HandRouter
from utils.normalization import LandmarkExtractor
from utils.cvfpscalc import CvFpsCalc
from utils.visualizer import Visualizer
from utils.data_collector import SequenceDataCollector
from model.one_hand_classifier.one_hand_classifier import OneHandClassifier
from model.two_hands_classifier.two_hands_classifier import TwoHandsClassifier


def get_args():
    parser = argparse.ArgumentParser(description="Hierarchical Dynamic Sign Language Recognition App")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=960, help="Camera width")
    parser.add_argument("--height", type=int, default=540, help="Camera height")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", type=float, default=0.80, help="Ngưỡng độ tin cậy hiển thị")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Thời gian làm mịn Debounce (giây)")
    parser.add_argument("--mode", type=int, default=1, choices=[0, 1], help="Chế độ khởi động: 0 (Suy luận), 1 (Thu thập dữ liệu, mặc định: 1)")
    parser.add_argument("--countdown_sec", type=float, default=2.0, help="Thời gian đếm ngược trước mẫu đầu tiên (giây)")
    parser.add_argument("--rest_sec", type=float, default=1.0, help="Thời gian nghỉ giữa các lần lấy mẫu (giây)")
    parser.add_argument("--samples", "--total_samples", dest="samples", type=int, default=40, help="Số lượng mẫu thu thập mỗi đợt bấm phím (mặc định: 40)")
    return parser.parse_args()


def load_branch_labels(branch_name):
    """
    Nạp danh sách nhãn cho từng nhánh từ tệp CSV tương ứng trong data/{branch_name}/
    """
    label_path = os.path.join("data", branch_name, "keypoint_classifier_label.csv")
    labels = []
    if os.path.exists(label_path):
        try:
            with open(label_path, encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                labels = [row[0].strip() for row in reader if row and row[0].strip()]
        except Exception as e:
            print(f"[WARNING] Không thể nạp nhãn cho nhánh '{branch_name}': {e}")

    if not labels:
        if branch_name == "one_hand":
            labels = ["Like", "OK", "Peace", "Point", "Stop"]
        else:
            labels = ["XinChao", "CamOn", "XinLoi", "YeuThuong", "TamBiet"]
    return labels


def main():
    args = get_args()

    # 1. Khởi tạo Camera
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # 2. Khởi tạo MediaPipe Holistic Pipeline
    holistic = mp_holistic.Holistic(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    # 3. Khởi tạo Các Module Hệ thống Phân nhánh
    router = HandRouter(history_len=5)
    extractor = LandmarkExtractor(use_3d=False)
    one_hand_clf = OneHandClassifier(model_path="model/one_hand_classifier/one_hand_classifier.tflite")
    two_hands_clf = TwoHandsClassifier(model_path="model/two_hands_classifier/two_hands_classifier.tflite")

    one_hand_labels = load_branch_labels("one_hand")
    two_hands_labels = load_branch_labels("two_hands")

    # Thu thập dữ liệu theo nhánh: mỗi lần 10 mẫu liên tục, nghỉ 1s giữa các mẫu (mặc định)
    collector_one_hand = SequenceDataCollector(
        data_dir="data/one_hand", sequence_length=30, countdown_sec=args.countdown_sec, rest_sec=args.rest_sec
    )
    collector_two_hands = SequenceDataCollector(
        data_dir="data/two_hands", sequence_length=30, countdown_sec=args.countdown_sec, rest_sec=args.rest_sec
    )

    cv_fps = CvFpsCalc(buffer_len=10)

    # Hàng đợi trượt 30 khung hình riêng biệt cho từng nhánh
    buf_one_hand = deque(maxlen=30)
    buf_two_hands = deque(maxlen=30)

    # Trạng thái ứng dụng
    mode = args.mode  # 0: Inference, 1: Data Collection (mặc định 1)
    current_batch_samples = args.samples  # Số lượng mẫu mỗi lần bấm phím (mặc định 40)
    branch_override = None  # None: Auto Router, "ONE_HAND": Ép 1 tay, "TWO_HANDS": Ép 2 tay
    collect_target_branch = "one_hand"  # "one_hand" hoặc "two_hands"
    collect_page = 0
    collect_page_size = 6

    # Trạng thái Debounce
    last_pred_time = 0.0
    last_pred_word = ""
    display_word = ""
    display_conf = 0.0
    display_branch = ""
    display_until = 0.0

    print("==========================================================")
    print(" HỆ THỐNG NHẬN DIỆN KÝ HIỆU PHÂN NHÁNH (ROUTER ARCHITECTURE)")
    print(" [ONE-HAND BRANCH]  : 58 đặc trưng (Thân trên + Bàn tay theo Mid-Shoulder)")
    print(" [TWO-HANDS BRANCH] : 100 đặc trưng (Thân trên + 2 Bàn tay theo Mid-Shoulder)")
    print("----------------------------------------------------------")
    print(" Phím [m] hoặc [Tab]: Chuyển chế độ Suy luận <-> Thu thập Dữ liệu")
    print(" Phím [b]: Đổi nhánh qua lại giữa 1 Tay và 2 Tay (hoạt động ở cả 2 mode)")
    print(" [Chế độ Suy luận / Mode 0]:")
    print("   - Phím [1]: Ép sang Nhánh 1 Tay")
    print("   - Phím [2]: Ép sang Nhánh 2 Tay")
    print("   - Phím [3] hoặc [a]: Chế độ Tự động (Auto Router)")
    print(" [Chế độ Thu thập Dữ liệu / Mode 1]:")
    print(f"   - Phím [0-5]: Thu thập {current_batch_samples} MẪU LIÊN TỤC cho nhãn trên trang hiện tại (Nghỉ {args.rest_sec}s/mẫu)")
    print("   - Phím [+]/[-]: Tăng / giảm số lượng mẫu mỗi đợt (ví dụ: 10, 20, 30, 40, ...)")
    print("   - Phím [n] / [p]: Lật trang danh sách từ (khi có nhiều hơn 6 từ)")
    print("   - Phím [b]: Đổi nhánh thu thập (1 Tay <-> 2 Tay)")
    print("   - Phím [c]: Hủy đợt thu thập đang chạy")
    print(" Phím [q / ESC]: Thoát")
    print("==========================================================")

    while cap.isOpened():
        fps = cv_fps.get()
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không thể đọc khung hình từ Camera.")
            break

        frame = cv.flip(frame, 1)
        debug_image = frame.copy()

        # 4. Xử lý MediaPipe Holistic
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True

        # Vẽ các điểm mốc Holistic
        Visualizer.draw_landmarks(debug_image, results)

        # 5. Định tuyến Nhánh (Router Determination)
        detected_branch = router.route(results)
        active_branch = branch_override if branch_override is not None else detected_branch

        # -------------------------------------------------------------
        # CHẾ ĐỘ 0: SUY LUẬN THỜI GIAN THỰC (REAL-TIME INFERENCE)
        # -------------------------------------------------------------
        if mode == 0:
            if active_branch == "ONE_HAND":
                # Lấy đặc trưng 58D (Thân trên + Bàn tay hoạt động)
                hand_lm = results.left_hand_landmarks or results.right_hand_landmarks
                if hand_lm is not None:
                    feat = extractor.normalize_one_hand(results, hand_landmarks=hand_lm)
                    buf_one_hand.append(feat)
                    buf_two_hands.clear()

                    if len(buf_one_hand) == 30 and one_hand_clf.is_loaded:
                        cid, conf, _ = one_hand_clf.predict(buf_one_hand)
                        if conf >= args.confidence_threshold:
                            word = one_hand_labels[cid] if cid < len(one_hand_labels) else f"Class_{cid}"
                            now = time.time()
                            if word != last_pred_word or (now - last_pred_time) > args.cooldown:
                                last_pred_word = word
                                last_pred_time = now
                                display_word = word
                                display_conf = conf
                                display_branch = "ONE_HAND"
                                display_until = now + 2.0
                else:
                    buf_one_hand.clear()

            elif active_branch == "TWO_HANDS":
                feat = extractor.normalize_two_hands(results)
                buf_two_hands.append(feat)
                buf_one_hand.clear()

                if len(buf_two_hands) == 30 and two_hands_clf.is_loaded:
                    cid, conf, _ = two_hands_clf.predict(buf_two_hands)
                    if conf >= args.confidence_threshold:
                        word = two_hands_labels[cid] if cid < len(two_hands_labels) else f"Class_{cid}"
                        now = time.time()
                        if word != last_pred_word or (now - last_pred_time) > args.cooldown:
                            last_pred_word = word
                            last_pred_time = now
                            display_word = word
                            display_conf = conf
                            display_branch = "TWO_HANDS"
                            display_until = now + 2.0
            else:
                buf_one_hand.clear()
                buf_two_hands.clear()

            # Vẽ Thẻ dự đoán nếu trong thời gian hiển thị
            if time.time() < display_until and display_word != "":
                Visualizer.draw_prediction(debug_image, display_word, display_conf, display_branch)

        # -------------------------------------------------------------
        # CHẾ ĐỘ 1: THU THẬP DỮ LIỆU (DATA COLLECTION MODE)
        # -------------------------------------------------------------
        elif mode == 1:
            if collect_target_branch == "one_hand":
                feat = extractor.normalize_one_hand(results)
                debug_image, _ = collector_one_hand.process_frame(feat, debug_image)
            else:
                feat = extractor.normalize_two_hands(results)
                debug_image, _ = collector_two_hands.process_frame(feat, debug_image)

        # -------------------------------------------------------------
        # VẼ GIAO DIỆN HỆ THỐNG (UI HEADER & FOOTER)
        # -------------------------------------------------------------
        current_labels = one_hand_labels if collect_target_branch == "one_hand" else two_hands_labels
        header_branch = ("ONE_HAND" if collect_target_branch == "one_hand" else "TWO_HANDS") if mode == 1 else active_branch
        header_is_manual = True if mode == 1 else (branch_override is not None)

        Visualizer.draw_header(debug_image, header_branch, mode, fps, is_manual=header_is_manual)
        Visualizer.draw_footer_instructions(
            debug_image,
            mode=mode,
            target_labels=current_labels,
            target_branch=collect_target_branch,
            page=collect_page,
            page_size=collect_page_size,
            samples_per_batch=current_batch_samples
        )

        cv.imshow("Hierarchical Sign Language Recognition", debug_image)

        # 6. Quản lý Phím Nhấn (Keyboard Controls)
        key = cv.waitKey(10) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('m'), ord('M'), 9, 32):  # m, M, Tab, Space
            collector_one_hand.cancel()
            collector_two_hands.cancel()
            mode = 1 - mode
            collect_page = 0
            buf_one_hand.clear()
            buf_two_hands.clear()
            print(f"[SYSTEM] Đã chuyển Mode -> {'[THU THẬP DỮ LIỆU / DATA COLLECTION]' if mode == 1 else '[SUY LUẬN / REAL-TIME INFERENCE]'}")
            if mode == 1:
                labels_str = ", ".join([f"[{i}]: {name}" for i, name in enumerate(current_labels)])
                print(f"[COLLECTOR] Nhánh hiện tại: [{collect_target_branch.upper()}]. Danh sách nhãn ({len(current_labels)} từ): {labels_str}")
                print("[COLLECTOR] Nhấn phím số (0-5) để thu thập, nhấn [n]/[p] để chuyển trang, hoặc nhấn [b] để đổi nhánh.")

        elif mode == 1:
            # --- CÁC PHÍM ĐIỀU KHIỂN TRONG CHẾ ĐỘ THU THẬP DỮ LIỆU ---
            if key in (ord('c'), ord('C')):
                collector_one_hand.cancel()
                collector_two_hands.cancel()
                print("[COLLECTOR] Đã dừng/hủy đợt thu thập.")
            elif key in (ord('b'), ord('B')):
                collector_one_hand.cancel()
                collector_two_hands.cancel()
                collect_page = 0
                if collect_target_branch == "one_hand":
                    collect_target_branch = "two_hands"
                    branch_override = "TWO_HANDS"
                else:
                    collect_target_branch = "one_hand"
                    branch_override = "ONE_HAND"
                cur_labels = one_hand_labels if collect_target_branch == "one_hand" else two_hands_labels
                labels_str = ", ".join([f"[{i}]: {name}" for i, name in enumerate(cur_labels)])
                print(f"[COLLECTOR] Đã chuyển sang Nhánh: [{collect_target_branch.upper()}] (data/{collect_target_branch}/)")
                print(f"[COLLECTOR] Danh sách nhãn: {labels_str}")
            elif key in (ord('n'), ord('N'), ord(']')):
                target_labels = one_hand_labels if collect_target_branch == "one_hand" else two_hands_labels
                total_pages = max(1, (len(target_labels) + collect_page_size - 1) // collect_page_size)
                collect_page = (collect_page + 1) % total_pages
                print(f"[COLLECTOR] Chuyển sang Trang [{collect_page + 1}/{total_pages}]")
            elif key in (ord('p'), ord('P'), ord('[')):
                target_labels = one_hand_labels if collect_target_branch == "one_hand" else two_hands_labels
                total_pages = max(1, (len(target_labels) + collect_page_size - 1) // collect_page_size)
                collect_page = (collect_page - 1 + total_pages) % total_pages
                print(f"[COLLECTOR] Chuyển sang Trang [{collect_page + 1}/{total_pages}]")
            elif key in (ord('+'), ord('=')):
                current_batch_samples = min(100, current_batch_samples + 10)
                print(f"[COLLECTOR] Đã đổi số lượng mẫu mỗi đợt: {current_batch_samples} mẫu/lần")
            elif key in (ord('-'), ord('_')):
                current_batch_samples = max(5, current_batch_samples - 10)
                print(f"[COLLECTOR] Đã đổi số lượng mẫu mỗi đợt: {current_batch_samples} mẫu/lần")
            elif ord('0') <= key <= ord('9'):
                slot = key - ord('0')
                target_labels = one_hand_labels if collect_target_branch == "one_hand" else two_hands_labels
                active_collector = collector_one_hand if collect_target_branch == "one_hand" else collector_two_hands

                idx = collect_page * collect_page_size + slot
                start_page_idx = collect_page * collect_page_size
                end_page_idx = min(start_page_idx + collect_page_size, len(target_labels))

                if idx < len(target_labels) and slot < (end_page_idx - start_page_idx):
                    sel_label = target_labels[idx]
                    success = active_collector.start_collection(
                        sel_label, total_samples=current_batch_samples, countdown_sec=args.countdown_sec, rest_sec=args.rest_sec
                    )
                    if success:
                        print(f"[COLLECTOR] Kích hoạt thu thập {current_batch_samples} mẫu liên tục cho nhãn '{sel_label}' (Chỉ số: {idx}) [{collect_target_branch}]")
                else:
                    print(f"[COLLECTOR] Phím [{slot}] không có nhãn trên Trang {collect_page + 1}. Dùng [n]/[p] để lật trang.")

        elif mode == 0:
            # --- CÁC PHÍM ĐIỀU KHIỂN TRONG CHẾ ĐỘ SUY LUẬN ---
            if key == ord('1'):
                branch_override = "ONE_HAND"
                collect_target_branch = "one_hand"
                buf_one_hand.clear()
                buf_two_hands.clear()
                print("[ROUTER] Ép chọn Nhánh: [1 TAY / ONE-HAND] (data/one_hand/)")
            elif key == ord('2'):
                branch_override = "TWO_HANDS"
                collect_target_branch = "two_hands"
                buf_one_hand.clear()
                buf_two_hands.clear()
                print("[ROUTER] Ép chọn Nhánh: [2 TAY / TWO-HANDS] (data/two_hands/)")
            elif key in (ord('3'), ord('a'), ord('A')):
                branch_override = None
                buf_one_hand.clear()
                buf_two_hands.clear()
                print("[ROUTER] Đã bật Chế độ: [TỰ ĐỘNG / AUTO ROUTER]")
            elif key in (ord('b'), ord('B')):
                if branch_override == "ONE_HAND":
                    branch_override = "TWO_HANDS"
                    collect_target_branch = "two_hands"
                else:
                    branch_override = "ONE_HAND"
                    collect_target_branch = "one_hand"
                buf_one_hand.clear()
                buf_two_hands.clear()
                print(f"[ROUTER] Chuyển Nhánh: [{branch_override}]")

    cap.release()
    cv.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    main()
