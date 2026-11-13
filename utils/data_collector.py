#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Thu thập Dữ liệu Chuỗi Thời gian (Data Collection) cho Ngôn ngữ Ký hiệu Động.
Quản lý quy trình Đếm ngược -> Thu thập 30 Khung hình -> Lưu file .npy.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import re
import time
import cv2 as cv
import numpy as np

from utils.visualizer import Visualizer


def sanitize_label_for_folder(label: str) -> str:
    """
    Chuẩn hóa tên nhãn để tạo thư mục an toàn trên hệ điều hành (đặc biệt Windows).
    Thay thế các ký tự cấm như /, \, :, *, ?, ", <, >, | thành '-'.
    Loại bỏ khoảng trắng hoặc dấu chấm ở đầu/cuối tên thư mục.
    """
    safe = re.sub(r'[\\/*?:"<>|]', '-', label.strip())
    safe = re.sub(r'\s*-\s*', ' - ', safe)
    safe = safe.strip('. ')
    return safe if safe else "unnamed_label"


class SequenceDataCollector:
    """
    Quản lý thu thập dữ liệu cử chỉ động theo lô (batch):
    Thu thập liên tục N mẫu (mặc định 10 mẫu), mỗi mẫu 30 khung hình, nghỉ 1 giây giữa mỗi mẫu.
    """

    STATE_IDLE = "IDLE"
    STATE_COUNTDOWN = "COUNTDOWN"  # Đếm ngược chuẩn bị hoặc nghỉ ngơi giữa các mẫu
    STATE_RECORDING = "RECORDING"  # Đang quay 30 khung hình cho 1 mẫu

    def __init__(self, data_dir="data", sequence_length=30, countdown_sec=2.0, rest_sec=1.0):
        """
        :param data_dir: Thư mục lưu dữ liệu thu thập
        :param sequence_length: Số lượng khung hình cố định cho 1 mẫu cử chỉ (default: 30)
        :param countdown_sec: Thời gian đếm ngược (giây) trước mẫu đầu tiên (default: 2.0s)
        :param rest_sec: Thời gian nghỉ (giây) giữa các mẫu liên tiếp (default: 1.0s)
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.countdown_sec = countdown_sec
        self.rest_sec = rest_sec

        self.state = self.STATE_IDLE
        self.current_label = ""
        self.sequence_buffer = []

        # Quản lý số lượng mẫu theo lô
        self.total_samples = 10
        self.current_sample_idx = 0
        self.current_countdown_duration = countdown_sec
        self.countdown_start_time = 0.0

        self.status_message = "CHẾ ĐỘ THU THẬP: Nhấn phím số để bắt đầu thu thập 10 mẫu liên tục."

        os.makedirs(self.data_dir, exist_ok=True)

    def start_collection(self, label_name, total_samples=10, countdown_sec=2.0, rest_sec=1.0):
        """
        Kích hoạt quy trình thu thập liên tục `total_samples` mẫu cho nhãn `label_name`.
        """
        if self.state != self.STATE_IDLE:
            print(f"[DATA COLLECTOR] Đang trong quá trình thu thập cho nhãn '{self.current_label}', vui lòng chờ hoặc nhấn [c] để hủy!")
            return False

        self.current_label = label_name
        self.total_samples = total_samples
        self.countdown_sec = countdown_sec
        self.rest_sec = rest_sec

        self.current_sample_idx = 1
        self.sequence_buffer = []
        self.current_countdown_duration = self.countdown_sec
        self.countdown_start_time = time.time()
        self.state = self.STATE_COUNTDOWN
        self.status_message = f"Chuẩn bị thu thập mẫu [1/{self.total_samples}] cho nhãn: [{self.current_label}]"
        print(f"[DATA COLLECTOR] Bắt đầu gói thu thập {self.total_samples} mẫu liên tục cho nhãn '{self.current_label}' (Nghỉ {self.rest_sec}s giữa mỗi mẫu).")
        return True

    def cancel(self):
        """
        Hủy bỏ đợt thu thập đang diễn ra.
        """
        if self.state != self.STATE_IDLE:
            print(f"[DATA COLLECTOR] Đã hủy đợt thu thập cho nhãn '{self.current_label}' (Đã hoàn thành {self.current_sample_idx - 1}/{self.total_samples} mẫu).")
            self.state = self.STATE_IDLE
            self.sequence_buffer = []
            self.status_message = "Đã hủy đợt thu thập."
            return True
        return False

    def process_frame(self, feature_vector, frame):
        """
        Cập nhật trạng thái thu thập và vẽ giao diện trực quan lên khung hình OpenCV.

        :param feature_vector: Vector đặc trưng từ LandmarkExtractor
        :param frame: Khung hình BGR từ camera
        :return: (frame_out, is_done) - Khung hình đã vẽ UI và cờ báo hoàn thành toàn bộ đợt
        """
        h, w, _ = frame.shape
        is_all_done = False

        if self.state == self.STATE_COUNTDOWN:
            elapsed = time.time() - self.countdown_start_time
            remaining = self.current_countdown_duration - elapsed

            if remaining > 0:
                # Vẽ hộp nền bán trong suốt ở giữa màn hình
                box_w, box_h = int(w * 0.7), 160
                box_x = int((w - box_w) / 2)
                box_y = int(h * 0.35)
                
                overlay = frame.copy()
                cv.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
                cv.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 2)

                # Tiêu đề trạng thái (Chuẩn bị mẫu đầu hoặc Nghỉ giữa các mẫu)
                if self.current_sample_idx == 1:
                    title_text = f"CHUẨN BỊ MẪU [1/{self.total_samples}]"
                    time_color = (0, 255, 255)
                else:
                    title_text = f"NGHỈ NGƠI -> MẪU TIẾP THEO [{self.current_sample_idx}/{self.total_samples}]"
                    time_color = (0, 200, 255)

                Visualizer.draw_texts(frame, [
                    (title_text, (box_x + 20, box_y + 15), 24, (255, 255, 255), True),
                    (f"BẮT ĐẦU SAU: {remaining:.1f}s", (box_x + 20, box_y + 55), 32, time_color, True),
                    (f"Nhãn: [{self.current_label}]  |  Nhấn [c] để hủy", (box_x + 20, box_y + 115), 18, (200, 200, 200), False)
                ])
            else:
                # Chuyển sang ghi hình cho mẫu hiện tại
                self.state = self.STATE_RECORDING
                self.sequence_buffer = []

        elif self.state == self.STATE_RECORDING:
            # Lưu vector đặc trưng vào buffer
            self.sequence_buffer.append(feature_vector)
            current_count = len(self.sequence_buffer)

            # Vẽ Thanh tiến trình (Progress Bar)
            progress = current_count / self.sequence_length
            bar_x, bar_y, bar_w, bar_h = int(w * 0.1), int(h * 0.82), int(w * 0.8), 35

            # Khung viền thanh tiến trình
            cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            # Thanh tiến trình chính (Màu xanh lá)
            fill_w = int(bar_w * progress)
            cv.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 0), -1)
            cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)

            # Hiển thị số khung hình và tiến độ mẫu tiếng Việt
            record_info = f"ĐANG QUAY [{self.current_label}] | Mẫu {self.current_sample_idx}/{self.total_samples} : {current_count}/{self.sequence_length}"
            Visualizer.draw_text(frame, record_info, (bar_x + 10, bar_y - 28), font_size=20, color=(0, 255, 0), is_bold=True)

            # Đã đủ 30 khung hình cho mẫu hiện tại -> Lưu file
            if current_count >= self.sequence_length:
                self.save_sequence()

                # Kiểm tra xem còn mẫu nào trong gói 10 mẫu không
                if self.current_sample_idx < self.total_samples:
                    # Chuyển sang nghỉ 3s trước khi sang mẫu kế tiếp
                    self.current_sample_idx += 1
                    self.sequence_buffer = []
                    self.current_countdown_duration = self.rest_sec
                    self.countdown_start_time = time.time()
                    self.state = self.STATE_COUNTDOWN
                    print(f"[DATA COLLECTOR] >> Xong mẫu {self.current_sample_idx - 1}/{self.total_samples}. Nghỉ {self.rest_sec}s chuẩn bị mẫu {self.current_sample_idx}/{self.total_samples}...")
                else:
                    # Đã hoàn thành đủ 10 mẫu
                    self.state = self.STATE_IDLE
                    self.sequence_buffer = []
                    is_all_done = True
                    self.status_message = f"HOAN THANH {self.total_samples} MAU CHO NHAN [{self.current_label}]!"
                    print(f"[DATA COLLECTOR] === HOÀN THÀNH TOÀN BỘ {self.total_samples} MẪU CHO NHÃN '{self.current_label}' ===")

        return frame, is_all_done

    def get_label_dir(self, label_name: str) -> str:
        """
        Lấy đường dẫn thư mục lưu trữ an toàn cho nhãn.
        """
        folder_name = sanitize_label_for_folder(label_name)
        return os.path.join(self.data_dir, folder_name)

    def save_sequence(self):
        """
        Lưu chuỗi 30 khung hình thành tệp NumPy (.npy).
        Cấu trúc thư mục: data/{branch}/{sanitized_label_name}/{timestamp}.npy
        """
        label_dir = self.get_label_dir(self.current_label)
        os.makedirs(label_dir, exist_ok=True)

        # Đặt tên file theo thời gian mili giây để tránh trùng lặp
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(label_dir, f"{timestamp}.npy")

        # Chuyển thành NumPy array dạng float32 (30, num_features)
        data_arr = np.array(self.sequence_buffer, dtype=np.float32)
        np.save(file_path, data_arr)

        sample_count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
        self.status_message = f"ĐÃ LƯU: Mau {self.current_sample_idx}/{self.total_samples} (Tong nhan [{self.current_label}]: {sample_count} mau)"
        print(f"[DATA COLLECTOR] Đã lưu: {file_path} (Tổng số mẫu trong nhãn: {sample_count})")
