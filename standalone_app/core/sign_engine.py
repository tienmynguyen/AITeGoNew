#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sign Language Recognition Engine (SignEngine) cho Ứng dụng Độc lập.
Tích hợp MediaPipe Holistic, Hand Router, và 2 Mô hình TFLite (1 Tay & 2 Tay).
Hỗ trợ cả 2 chế độ:
  1. Camera Server: OpenCV VideoCapture đa luồng (MJPEG stream).
  2. Camera Trình duyệt / WebCam: Nhận frame qua WebSocket nhị phân.
"""

import os
import sys
import time
import csv
import threading
from collections import deque
from typing import Optional, Callable, Dict, Any, List

import cv2 as cv
import numpy as np

# MediaPipe Solutions
try:
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic

from .hand_router import HandRouter
from .normalization import LandmarkExtractor
from .cvfpscalc import CvFpsCalc
from .visualizer import Visualizer
from .classifiers import OneHandClassifier, TwoHandsClassifier

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(APP_DIR, "models")


def load_labels(branch_name: str) -> List[str]:
    """Đọc nhãn từ tệp CSV tương ứng trong thư mục models/."""
    label_path = os.path.join(MODELS_DIR, f"labels_{branch_name}.csv")
    labels = []
    if os.path.exists(label_path):
        try:
            with open(label_path, encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                labels = [row[0].strip() for row in reader if row and row[0].strip()]
        except Exception as e:
            print(f"[WARNING] Không thể nạp nhãn {branch_name}: {e}")

    if not labels:
        if branch_name == "one_hand":
            labels = ['xin chào', 'tốt', 'tôi', 'bạn', 'chúng tôi', 'bọn họ', 'có / đúng', 'không', 'yêu', 'dừng lại']
        else:
            labels = ['cảm ơn', 'xin lỗi', 'aaa', 'làm ơn', 'giá', 'không', 'cái gì', 'ở đâu', 'khi nào', 'ai', 'tại sao', 'bao nhiêu', 'muốn', 'giúp', 'tạm biệt', 'tên', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '00', '000']
    return labels


class SignRecognitionEngine:
    """
    Động cơ nhận diện cử chỉ ngôn ngữ ký hiệu phân nhánh thời gian thực.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        confidence_threshold: float = 0.68,
        cooldown: float = 1.0,
        draw_landmarks: bool = True,
        hold_duration: float = 1.0
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown
        self.draw_landmarks = draw_landmarks
        self.hold_duration = hold_duration

        # Nhãn & Mô hình
        self.one_hand_labels = load_labels("one_hand")
        self.two_hands_labels = load_labels("two_hands")

        one_hand_model_path = os.path.join(MODELS_DIR, "one_hand_classifier.tflite")
        two_hands_model_path = os.path.join(MODELS_DIR, "two_hands_classifier.tflite")

        self.one_hand_clf = OneHandClassifier(model_path=one_hand_model_path)
        self.two_hands_clf = TwoHandsClassifier(model_path=two_hands_model_path)

        self.router = HandRouter(history_len=5)
        self.extractor = LandmarkExtractor(use_3d=False)

        # Chế độ hoạt động
        self.branch_override: Optional[str] = None  # None: Auto, "ONE_HAND", "TWO_HANDS"
        self.active_branch: str = "NO_HAND"

        # Bộ đệm chuỗi khung hình trượt
        self.buf_one_hand = deque(maxlen=30)
        self.buf_two_hands = deque(maxlen=30)
        self.no_hand_counter = 0

        # Trạng thái Duy trì cử chỉ (Hold Duration: 1.0s) & Bộ đệm chống rung
        self.current_hold_word = ""
        self.current_hold_start = 0.0
        self.current_hold_branch = ""
        self.current_hold_conf = 0.0
        self.hold_confirmed = False
        self.mismatch_frames = 0
        self.MAX_MISMATCH_FRAMES = 8   # Dung sai 8 khung hình (~250ms) chống rung tay khi giữ 1.0s
        self.last_seen_word_time = 0.0
        self.miss_grace_period = 0.35  # Thời gian ân hạn rớt frame
        self.hold_grace_period = 0.35  # Thời gian ân hạn duy trì cử chỉ

        # Trạng thái Debounce & Hiển thị
        self.last_pred_word = ""
        self.last_pred_time = 0.0
        self.display_word = ""
        self.display_conf = 0.0
        self.display_branch = ""
        self.display_until = 0.0

        # MediaPipe Holistic Pipeline dùng chung
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Quản lý Camera & Thread OpenCV Server
        self.cap = None
        self.is_running = False
        self.server_cam_available = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.fps = 0.0
        self.cv_fps = CvFpsCalc(buffer_len=10)

        # Danh sách callback khi có từ mới được nhận diện
        self.word_callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def register_word_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Đăng ký hàm callback nhận sự kiện từ mới."""
        if callback not in self.word_callbacks:
            self.word_callbacks.append(callback)

    def set_branch_override(self, mode_str: str):
        """Đặt chế độ phân nhánh: 'AUTO', 'ONE_HAND', 'TWO_HANDS'."""
        mode_str = mode_str.upper()
        with self.lock:
            if mode_str in ("ONE_HAND", "TWO_HANDS"):
                self.branch_override = mode_str
            else:
                self.branch_override = None
            self.buf_one_hand.clear()
            self.buf_two_hands.clear()

    def set_settings(
        self,
        confidence_threshold: Optional[float] = None,
        cooldown: Optional[float] = None,
        draw_landmarks: Optional[bool] = None,
        hold_duration: Optional[float] = None
    ):
        """Cập nhật các thông số suy luận."""
        with self.lock:
            if confidence_threshold is not None:
                self.confidence_threshold = float(confidence_threshold)
            if cooldown is not None:
                self.cooldown = float(cooldown)
            if draw_landmarks is not None:
                self.draw_landmarks = bool(draw_landmarks)
            if hold_duration is not None:
                self.hold_duration = max(0.2, float(hold_duration))

    @staticmethod
    def _pad_sequence(buf_seq, target_len: int = 30) -> list:
        """Đệm chuỗi khung hình từ độ dài hiện tại lên 30."""
        seq_list = list(buf_seq)
        if len(seq_list) < target_len:
            seq_list = seq_list + [seq_list[-1]] * (target_len - len(seq_list))
        return seq_list

    def process_frame(self, frame: np.ndarray, return_annotated: bool = False) -> Dict[str, Any]:
        """
        Xử lý 1 khung hình (từ browser webcam hoặc OpenCV server).
        Đảm bảo an toàn đa luồng.
        """
        with self.lock:
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            # Định tuyến nhánh tự động
            detected_branch = self.router.route(results)
            active_branch = self.branch_override if self.branch_override is not None else detected_branch
            self.active_branch = active_branch

            recognized_word = ""
            conf = 0.0
            word_branch = ""
            word_event = None
            now = time.time()
            MIN_FRAMES = 6

            if active_branch == "ONE_HAND":
                hand_lm = results.left_hand_landmarks or results.right_hand_landmarks
                if hand_lm is not None:
                    self.no_hand_counter = 0
                    feat_1h = self.extractor.normalize_one_hand(results, hand_landmarks=hand_lm)
                    self.buf_one_hand.append(feat_1h)
                    self.buf_two_hands.clear()

                    if len(self.buf_one_hand) >= MIN_FRAMES and self.one_hand_clf.is_loaded:
                        padded_1h = self._pad_sequence(self.buf_one_hand)
                        cid_1h, conf_1h, _ = self.one_hand_clf.predict(padded_1h)
                        if conf_1h >= self.confidence_threshold:
                            word = self.one_hand_labels[cid_1h] if cid_1h < len(self.one_hand_labels) else f"Class_{cid_1h}"
                            recognized_word = word
                            conf = float(conf_1h)
                            word_branch = "ONE_HAND"
                else:
                    self.no_hand_counter += 1
                    if self.no_hand_counter >= 8:
                        self.buf_one_hand.clear()

            elif active_branch == "TWO_HANDS":
                if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
                    self.no_hand_counter = 0
                    feat_2h = self.extractor.normalize_two_hands(results)
                    self.buf_two_hands.append(feat_2h)
                    self.buf_one_hand.clear()

                    if len(self.buf_two_hands) >= MIN_FRAMES and self.two_hands_clf.is_loaded:
                        padded_2h = self._pad_sequence(self.buf_two_hands)
                        cid_2h, conf_2h, _ = self.two_hands_clf.predict(padded_2h)
                        if conf_2h >= self.confidence_threshold:
                            word = self.two_hands_labels[cid_2h] if cid_2h < len(self.two_hands_labels) else f"Class_{cid_2h}"
                            recognized_word = word
                            conf = float(conf_2h)
                            word_branch = "TWO_HANDS"
                else:
                    self.no_hand_counter += 1
                    if self.no_hand_counter >= 8:
                        self.buf_two_hands.clear()

            else:
                self.no_hand_counter += 1
                if self.no_hand_counter >= 8:
                    self.buf_one_hand.clear()
                    self.buf_two_hands.clear()

            # Cơ chế Duy trì cử chỉ (Hold Duration 2.0s) & Chống Rung (Jitter Cushion)
            hold_progress = 0.0
            hold_seconds = 0.0

            if recognized_word:
                self.last_seen_word_time = now
                if not self.current_hold_word:
                    self.current_hold_word = recognized_word
                    self.current_hold_start = now
                    self.current_hold_branch = word_branch
                    self.current_hold_conf = conf
                    self.hold_confirmed = False
                    self.mismatch_frames = 0
                    hold_seconds = 0.0
                    hold_progress = 0.0
                elif recognized_word == self.current_hold_word:
                    self.mismatch_frames = 0
                    self.current_hold_conf = conf
                    self.current_hold_branch = word_branch
                    hold_seconds = max(0.0, now - self.current_hold_start)
                    hold_progress = min(1.0, hold_seconds / self.hold_duration)

                    if hold_seconds >= self.hold_duration and not self.hold_confirmed:
                        self.hold_confirmed = True
                        self.last_pred_word = recognized_word
                        self.last_pred_time = now
                        self.display_word = recognized_word
                        self.display_conf = conf
                        self.display_branch = word_branch
                        self.display_until = now + 2.5

                        word_event = {
                            "type": "word_recognized",
                            "word": recognized_word,
                            "confidence": round(conf, 3),
                            "branch": word_branch,
                            "timestamp": now,
                            "hold_seconds": round(hold_seconds, 2)
                        }
                    elif self.hold_confirmed:
                        hold_progress = 1.0
                else:
                    self.mismatch_frames += 1
                    if self.mismatch_frames >= self.MAX_MISMATCH_FRAMES:
                        self.current_hold_word = recognized_word
                        self.current_hold_start = now
                        self.current_hold_branch = word_branch
                        self.current_hold_conf = conf
                        self.hold_confirmed = False
                        self.mismatch_frames = 0
                        hold_seconds = 0.0
                        hold_progress = 0.0
                    else:
                        hold_seconds = max(0.0, now - self.current_hold_start)
                        hold_progress = min(1.0, hold_seconds / self.hold_duration)
            else:
                if self.current_hold_word:
                    self.mismatch_frames += 1
                    if self.mismatch_frames >= self.MAX_MISMATCH_FRAMES or (now - self.last_seen_word_time > self.hold_grace_period):
                        self.current_hold_word = ""
                        self.current_hold_start = 0.0
                        self.current_hold_branch = ""
                        self.current_hold_conf = 0.0
                        self.hold_confirmed = False
                        self.mismatch_frames = 0
                        hold_seconds = 0.0
                        hold_progress = 0.0
                    else:
                        hold_seconds = max(0.0, now - self.current_hold_start)
                        hold_progress = min(1.0, hold_seconds / self.hold_duration)

            landmarks_dict = self._extract_compact_landmarks(results)

            annotated_jpeg = None
            if return_annotated:
                debug_img = frame.copy()
                if self.draw_landmarks:
                    Visualizer.draw_landmarks(debug_img, results)
                active_display_word = self.current_hold_word or (self.display_word if now < self.display_until else "")
                active_display_conf = self.current_hold_conf if self.current_hold_word else (self.display_conf if now < self.display_until else 0.0)
                active_display_branch = self.current_hold_branch or self.display_branch
                if active_display_word:
                    current_hold_info = {
                        "seconds": hold_seconds,
                        "target": self.hold_duration,
                        "progress": hold_progress,
                        "confirmed": self.hold_confirmed
                    }
                    Visualizer.draw_prediction(
                        debug_img,
                        active_display_word,
                        active_display_conf,
                        active_display_branch,
                        hold_info=current_hold_info
                    )
                Visualizer.draw_header(debug_img, active_branch, mode=0, fps=self.fps, is_manual=(self.branch_override is not None))
                _, buf = cv.imencode('.jpg', debug_img, [int(cv.IMWRITE_JPEG_QUALITY), 80])
                annotated_jpeg = buf.tobytes()

            if word_event:
                for cb in list(self.word_callbacks):
                    try:
                        cb(word_event)
                    except Exception as e:
                        print(f"[SIGN_ENGINE] Callback error: {e}")

            return {
                "type": "frame_result",
                "active_branch": active_branch,
                "word": recognized_word or (self.current_hold_word if (now - self.last_seen_word_time <= self.miss_grace_period) else (self.display_word if now < self.display_until else "")),
                "confidence": round(conf if conf > 0 else (self.current_hold_conf if self.current_hold_word else (self.display_conf if now < self.display_until else 0.0)), 2),
                "branch_override": self.branch_override,
                "landmarks": landmarks_dict,
                "word_event": word_event,
                "holding": {
                    "word": self.current_hold_word,
                    "branch": self.current_hold_branch,
                    "seconds": round(hold_seconds, 2),
                    "target": round(self.hold_duration, 2),
                    "progress": round(hold_progress, 2),
                    "confirmed": self.hold_confirmed
                },
                "annotated_jpeg": annotated_jpeg
            }

    def _extract_compact_landmarks(self, results) -> Dict[str, Any]:
        """Trích xuất tọa độ điểm mốc (x, y) chuẩn hóa 0..1 để client vẽ canvas."""
        res = {"pose": [], "left_hand": [], "right_hand": []}
        if results.pose_landmarks:
            for idx in [0, 11, 12, 13, 14, 15, 16]:
                lm = results.pose_landmarks.landmark[idx]
                res["pose"].append({"i": idx, "x": round(lm.x, 3), "y": round(lm.y, 3)})

        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                res["left_hand"].append({"i": i, "x": round(lm.x, 3), "y": round(lm.y, 3)})

        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                res["right_hand"].append({"i": i, "x": round(lm.x, 3), "y": round(lm.y, 3)})

        return res

    def start(self, device_id: Optional[int] = None) -> bool:
        """Khởi động camera server (OpenCV)."""
        if self.is_running:
            return True

        if device_id is not None:
            self.device_id = device_id

        cap = cv.VideoCapture(self.device_id)
        if not cap.isOpened():
            cap.release()
            cap = cv.VideoCapture(self.device_id, cv.CAP_DSHOW)

        if not cap.isOpened():
            print(f"[SIGN_ENGINE] Không tìm thấy Camera phần cứng ID {self.device_id} trên máy chủ.")
            self.server_cam_available = False
            return False

        self.cap = cap
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.is_running = True
        self.server_cam_available = True

        self.thread = threading.Thread(target=self._run_server_cam_loop, daemon=True)
        self.thread.start()
        print(f"[SIGN_ENGINE] Camera Server {self.device_id} đã khởi động thành công.")
        return True

    def stop(self):
        """Dừng camera server."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.latest_jpeg = None
        print("[SIGN_ENGINE] Đã dừng camera server.")

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hiện tại của engine."""
        now = time.time()
        hold_sec = max(0.0, now - self.current_hold_start) if self.current_hold_word else 0.0
        hold_prog = min(1.0, hold_sec / self.hold_duration) if self.current_hold_word else 0.0
        return {
            "is_running": self.is_running,
            "server_cam_available": self.server_cam_available,
            "device_id": self.device_id,
            "fps": round(self.fps, 1),
            "active_branch": self.active_branch,
            "branch_override": self.branch_override,
            "confidence_threshold": self.confidence_threshold,
            "cooldown": self.cooldown,
            "draw_landmarks": self.draw_landmarks,
            "hold_duration": self.hold_duration,
            "buf_one_hand_len": len(self.buf_one_hand),
            "buf_two_hands_len": len(self.buf_two_hands),
            "last_word": self.display_word,
            "last_confidence": round(self.display_conf, 2),
            "one_hand_labels_count": len(self.one_hand_labels),
            "two_hands_labels_count": len(self.two_hands_labels),
            "holding": {
                "word": self.current_hold_word,
                "branch": self.current_hold_branch,
                "seconds": round(hold_sec, 2),
                "target": round(self.hold_duration, 2),
                "progress": round(hold_prog, 2),
                "confirmed": self.hold_confirmed
            }
        }

    def _run_server_cam_loop(self):
        """Vòng lặp đọc khung hình từ OpenCV server camera."""
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                self.fps = self.cv_fps.get()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv.flip(frame, 1)
                res = self.process_frame(frame, return_annotated=True)
                with self.lock:
                    self.latest_jpeg = res.get("annotated_jpeg")
            except Exception as e:
                print(f"[SERVER CAM LOOP EXCEPTION] {e}")

            time.sleep(0.005)

