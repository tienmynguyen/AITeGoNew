#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Trực quan hóa Giao diện (Visualizer).
Quản lý vẽ UI, Bounding Box, Thẻ kết quả dự đoán, Trạng thái Nhánh (Branch Status) và Thanh tiến trình.
"""

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from mediapipe.python.solutions import holistic as mp_holistic
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils


class Visualizer:
    """
    Lớp vẽ giao diện đồ họa OpenCV trực quan cho Hệ thống Nhận diện Ngôn ngữ Ký hiệu Phân nhánh.
    Hỗ trợ hiển thị tiếng Việt có dấu đầy đủ qua thư viện Pillow TrueType Font.
    """

    COLOR_ONE_HAND = (255, 255, 0)      # Màu Xanh Cyan cho Nhánh 1 tay
    COLOR_TWO_HANDS = (255, 0, 255)     # Màu Tím Magenta cho Nhánh 2 tay
    COLOR_NO_HAND = (150, 150, 150)     # Màu Xám cho Trạng thái không có tay
    COLOR_BG = (30, 30, 30)             # Màu nền xám đậm

    _FONT_CACHE = {}

    @classmethod
    def get_font(cls, size, is_bold=False):
        """
        Nạp font TrueType hỗ trợ tiếng Việt Unicode từ Windows hoặc Linux.
        """
        key = (size, is_bold)
        if key not in cls._FONT_CACHE:
            font = None
            if is_bold:
                candidates = [
                    "C:/Windows/Fonts/arialbd.ttf",
                    "C:/Windows/Fonts/segoeuib.ttf",
                    "C:/Windows/Fonts/tahomabd.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
                ]
            else:
                candidates = [
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/segoeui.ttf",
                    "C:/Windows/Fonts/tahoma.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ]
            for p in candidates:
                if os.path.exists(p):
                    try:
                        font = ImageFont.truetype(p, size)
                        break
                    except Exception:
                        pass
            if font is None:
                font = ImageFont.load_default()
            cls._FONT_CACHE[key] = font
        return cls._FONT_CACHE[key]

    @classmethod
    def draw_texts(cls, image, text_items):
        """
        Vẽ danh sách các chuỗi văn bản tiếng Việt Unicode lên ảnh BGR trong 1 lần chuyển đổi PIL.
        :param image: numpy array BGR
        :param text_items: List các tuple (text, (x, y), font_size, bgr_color, is_bold)
        """
        if not text_items:
            return image

        pil_img = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for item in text_items:
            text = str(item[0]) if item[0] is not None else ""
            if not text:
                continue
            pos = item[1]
            size = item[2] if len(item) > 2 else 16
            color = item[3] if len(item) > 3 else (255, 255, 255)
            is_bold = item[4] if len(item) > 4 else False

            b, g, r = color[:3]
            rgb_color = (int(r), int(g), int(b))
            font = cls.get_font(size, is_bold=is_bold)
            draw.text(pos, text, font=font, fill=rgb_color)

        image[:] = cv.cvtColor(np.asarray(pil_img), cv.COLOR_RGB2BGR)
        return image

    @classmethod
    def draw_text(cls, image, text, position, font_size=16, color=(255, 255, 255), is_bold=False):
        """
        Vẽ 1 chuỗi văn bản tiếng Việt Unicode lên ảnh BGR.
        """
        return cls.draw_texts(image, [(text, position, font_size, color, is_bold)])

    @staticmethod
    def draw_landmarks(image, results):
        """
        Vẽ các điểm mốc MediaPipe Holistic (Pose + 2 Bàn tay).
        """
        if results is None:
            return

        # 1. Vẽ Pose (Thân trên)
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2)
            )

        # 2. Vẽ Tay trái
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
            )

        # 3. Vẽ Tay phải
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )

    @classmethod
    def draw_header(cls, image, active_branch, mode, fps, is_manual=False):
        """
        Vẽ thanh tiêu đề hệ thống ở cạnh trên khung hình bằng tiếng Việt có dấu.

        :param active_branch: "ONE_HAND", "TWO_HANDS", hoặc "NO_HAND"
        :param mode: 0 (Inference) hoặc 1 (Collection)
        :param fps: Tốc độ khung hình / giây
        :param is_manual: True nếu đang ép nhánh thủ công, False nếu tự động
        """
        h, w, _ = image.shape
        cv.rectangle(image, (0, 0), (w, 45), cls.COLOR_BG, -1)

        tag = "(THỦ CÔNG)" if is_manual else "(TỰ ĐỘNG)"

        # Xác định màu sắc & nhãn nhánh
        if active_branch == "ONE_HAND":
            branch_text = f"NHÁNH: [1 TAY / ONE-HAND] {tag}"
            branch_color = cls.COLOR_ONE_HAND
        elif active_branch == "TWO_HANDS":
            branch_text = f"NHÁNH: [2 TAY / TWO-HANDS] {tag}"
            branch_color = cls.COLOR_TWO_HANDS
        else:
            branch_text = f"NHÁNH: [CHƯA PHÁT HIỆN TAY] {tag}"
            branch_color = cls.COLOR_NO_HAND

        mode_text = "SUY LUẬN THỜI GIAN THỰC" if mode == 0 else "THU THẬP DỮ LIỆU"
        left_text = f"CHẾ ĐỘ: {mode_text} | FPS: {fps}"

        font = cls.get_font(16, is_bold=True)
        bbox = font.getbbox(branch_text)
        branch_w = bbox[2] - bbox[0]

        cls.draw_texts(image, [
            (left_text, (15, 12), 16, (255, 255, 255), True),
            (branch_text, (w - branch_w - 20, 12), 16, branch_color, True)
        ])

    @classmethod
    def draw_prediction(cls, image, word, confidence, branch_name, hold_info=None):
        """
        Vẽ thẻ hiển thị kết quả nhận diện và thanh tiến trình giữ 1s ở góc dưới bằng tiếng Việt.
        """
        if not word:
            return

        h, w, _ = image.shape
        badge_color = cls.COLOR_ONE_HAND if branch_name == "ONE_HAND" else cls.COLOR_TWO_HANDS
        branch_title = "1 TAY" if branch_name == "ONE_HAND" else "2 TAY"

        word_text = f"TỪ [{branch_title}]: {word.upper()}"
        conf_text = f"Độ tin cậy: {confidence * 100:.0f}%"

        font_word = cls.get_font(20, is_bold=True)
        bbox = font_word.getbbox(word_text)
        word_w = bbox[2] - bbox[0]
        box_w = max(440, word_w + 60)

        # Trạng thái giữ cử chỉ (Hold progress)
        has_hold = hold_info is not None and isinstance(hold_info, dict)
        card_h = 135 if has_hold else 90
        top_y = h - card_h - 15
        bot_y = h - 15

        # Nền thẻ bo góc đen viền màu nhánh
        cv.rectangle(image, (20, top_y), (20 + box_w, bot_y), (12, 12, 12), -1)
        cv.rectangle(image, (20, top_y), (20 + box_w, bot_y), badge_color, 2)

        text_items = [
            (word_text, (35, top_y + 10), 20, badge_color, True),
            (conf_text, (20 + box_w - 150, top_y + 12), 15, (200, 200, 200), False)
        ]

        if has_hold:
            sec = hold_info.get("seconds", 0.0)
            target = hold_info.get("target", 1.0)
            prog = min(1.0, max(0.0, hold_info.get("progress", 0.0)))
            confirmed = hold_info.get("confirmed", False)

            # Vẽ thanh tiến trình Hold Bar
            bar_x = 35
            bar_y = top_y + 48
            bar_w = box_w - 30
            bar_h = 14

            # Nền thanh (xám đậm)
            cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
            cv.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

            # Phần lấp đầy tiến trình
            fill_w = int(bar_w * (1.0 if confirmed else prog))
            fill_color = (0, 230, 120) if confirmed else (255, 180, 0)  # Xanh lục nếu chốt, Vàng/Cyan nếu đang giữ
            if fill_w > 0:
                cv.rectangle(image, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

            if confirmed:
                status_text = f"✅ ĐÃ CHỐT VÀO HÀNG ĐỢI!"
                status_color = (0, 240, 120)
            else:
                pct = int(prog * 100)
                status_text = f"⏳ Giữ cử chỉ: {sec:.1f}s / {target:.1f}s ({pct}%)"
                status_color = (220, 220, 220)

            text_items.append((status_text, (35, bar_y + 20), 16, status_color, True))

        cls.draw_texts(image, text_items)

    @classmethod
    def draw_footer_instructions(cls, image, mode=0, target_labels=None, target_branch="one_hand", page=0, page_size=6, samples_per_batch=40):
        """
        Vẽ dòng hướng dẫn phím bấm và danh sách nhãn tiếng Việt ở chân khung hình.
        """
        h, w, _ = image.shape
        footer_height = 65 if (mode == 1 and target_labels) else 38
        cv.rectangle(image, (0, h - footer_height), (w, h), cls.COLOR_BG, -1)

        if mode == 0:
            footer = "[m]/[Tab]: Chuyển sang Thu Thập | [1]: Ép 1 Tay | [2]: Ép 2 Tay | [3]/[a]: Tự Động | [b]: Đổi Nhánh | [q]: Thoát"
            cls.draw_text(image, footer, (15, h - 26), font_size=15, color=(220, 220, 220), is_bold=False)
        else:
            branch_label = "1 TAY" if target_branch == "one_hand" else "2 TAY"
            footer = f"[m]: Về Suy Luận | [b]: Đổi Nhánh ({branch_label}) | [+/-]: {samples_per_batch} mẫu/đợt | [c]: Hủy | [q]: Thoát"

            text_items = []
            if target_labels:
                total_labels = len(target_labels)
                total_pages = max(1, (total_labels + page_size - 1) // page_size)
                curr_page = max(0, min(page, total_pages - 1))
                start_idx = curr_page * page_size
                end_idx = min(start_idx + page_size, total_labels)
                page_labels = target_labels[start_idx:end_idx]

                labels_str = "    ".join([f"[{i}]: {name}" for i, name in enumerate(page_labels)])
                if total_pages > 1:
                    page_info = f"Trang [{curr_page + 1}/{total_pages}] (Phím [n]/[p] lật trang):  "
                    label_line = page_info + labels_str
                else:
                    label_line = f"Nhấn phím số để thu thập:  " + labels_str

                text_items.append((label_line, (15, h - 56), 15, (0, 255, 255), True))
                text_items.append((footer, (15, h - 28), 14, (200, 200, 200), False))
            else:
                text_items.append((footer, (15, h - 26), 14, (200, 200, 200), False))

            cls.draw_texts(image, text_items)

