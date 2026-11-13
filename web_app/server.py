#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web Server cho Hệ thống Nhận diện Ký hiệu & Ghép câu Gemini LLM.
Sử dụng Tornado Web Server hỗ trợ:
  - MJPEG video streaming an toàn
  - WebSocket hai chiều (hỗ trợ truyền frame nhị phân từ Browser Webcam và trả kết quả nhận diện)
  - REST API cho Gemini LLM và điều khiển hệ thống
"""

import os
import sys
import json
import asyncio
import tempfile
import argparse

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_vsl')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.httpserver
import tornado.iostream
import cv2 as cv
import numpy as np

from web_app.sign_engine import SignRecognitionEngine
from web_app.gemini_service import GeminiSentenceBuilder, SUPPORTED_MODELS, DEFAULT_GEMINI_MODEL


# Khởi tạo các Singletons
engine = SignRecognitionEngine()
gemini_builder = GeminiSentenceBuilder()

# Lưu trữ danh sách kết nối WebSocket
ws_clients = set()


class MainHandler(tornado.web.RequestHandler):
    """Trang chủ ứng dụng."""
    def get(self):
        self.render("index.html")


class VideoFeedHandler(tornado.web.RequestHandler):
    """
    Luồng MJPEG Video Stream thời gian thực từ OpenCV Server Camera.
    Tương thích hoàn toàn với Tornado 6+, xử lý đóng kết nối an toàn.
    """
    async def get(self):
        self.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.set_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.set_header('Pragma', 'no-cache')
        self.set_header('Expires', '0')

        blank_frame = None

        try:
            while True:
                # Kiểm tra kết nối đóng an toàn
                conn = self.request.connection
                if hasattr(conn, 'stream') and conn.stream and conn.stream.closed():
                    break

                jpeg_bytes = engine.get_latest_jpeg()
                if jpeg_bytes is None:
                    if blank_frame is None:
                        # Ảnh thông báo chờ kết nối
                        img = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv.putText(img, "Dang cho Camera OpenCV...", (160, 160), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        cv.putText(img, "(Khuyen dung: Chon che do 'Camera Trinh Duyet')", (100, 210), cv.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)
                        _, buf = cv.imencode('.jpg', img)
                        blank_frame = buf.tobytes()
                    jpeg_bytes = blank_frame

                self.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
                await self.flush()
                await asyncio.sleep(0.04)  # ~25 FPS

        except (tornado.iostream.StreamClosedError, asyncio.CancelledError, Exception):
            pass


class RealtimeWebSocket(tornado.websocket.WebSocketHandler):
    """
    WebSocket thời gian thực:
      - Nhận frame nhị phân (Binary JPEG) từ Browser Webcam, xử lý và trả kết quả tức thì.
      - Trao đổi cấu hình, trạng thái, và phát hiện từ mới.
    """
    def check_origin(self, origin):
        return True

    def open(self):
        global MAIN_IOLOOP
        MAIN_IOLOOP = tornado.ioloop.IOLoop.current()
        ws_clients.add(self)
        init_data = {
            "type": "init",
            "status": engine.get_status(),
            "labels": {
                "one_hand": engine.one_hand_labels,
                "two_hands": engine.two_hands_labels
            },
            "gemini": {
                "has_key": bool(gemini_builder.api_key),
                "model": gemini_builder.model,
                "supported_models": SUPPORTED_MODELS
            }
        }
        self.write_message(json.dumps(init_data, ensure_ascii=False))

    def on_close(self):
        ws_clients.discard(self)

    def on_message(self, message):
        # 1. Xử lý frame nhị phân gửi từ Browser Webcam (canvas.toBlob)
        if isinstance(message, bytes):
            try:
                nparr = np.frombuffer(message, np.uint8)
                frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
                if frame is not None:
                    # Lật gương khung hình để chuẩn xác với tư thế nhìn camera selfie (đồng bộ 100% với Server Camera)
                    frame = cv.flip(frame, 1)
                    # Pipeline nhận diện cử chỉ
                    result = engine.process_frame(frame, return_annotated=False)
                    self.write_message(json.dumps(result, ensure_ascii=False))
            except Exception as e:
                print(f"[WS BINARY ERROR] {e}")
            return

        # 2. Xử lý tin nhắn JSON điều khiển
        try:
            data = json.loads(message)
            action = data.get("action")

            if action == "set_branch":
                mode = data.get("mode", "AUTO")
                engine.set_branch_override(mode)
                self.broadcast_status()

            elif action == "toggle_camera":
                state = data.get("state")
                dev_id = data.get("device_id", 0)
                if state:
                    engine.start(device_id=dev_id)
                else:
                    engine.stop()
                self.broadcast_status()

            elif action == "set_settings":
                conf = data.get("confidence_threshold")
                cd = data.get("cooldown")
                draw_lm = data.get("draw_landmarks")
                hold_dur = data.get("hold_duration")
                engine.set_settings(
                    confidence_threshold=conf,
                    cooldown=cd,
                    draw_landmarks=draw_lm,
                    hold_duration=hold_dur
                )
                self.broadcast_status()

        except Exception as e:
            print(f"[WS JSON ERROR] {e}")

    @classmethod
    def broadcast_status(cls):
        status = {"type": "status", "status": engine.get_status()}
        msg = json.dumps(status, ensure_ascii=False)
        for client in list(ws_clients):
            try:
                client.write_message(msg)
            except Exception:
                pass


# Quản lý Event Loop chính cho điều phối đa luồng an toàn
try:
    MAIN_IOLOOP = tornado.ioloop.IOLoop.current()
except Exception:
    MAIN_IOLOOP = None


def on_engine_word_recognized(event: dict):
    """Callback chuyển tiếp sự kiện từ mới tới tất cả các WebSocket client (an toàn đa luồng)."""
    word = event.get("word")
    branch = event.get("branch")
    conf = event.get("confidence")
    print(f"[SERVER EVENT] >>> CHỐT TỪ: '{word}' ({branch}, conf={conf}) -> Đang gửi tới {len(ws_clients)} WebSocket client(s)...")
    msg = json.dumps(event, ensure_ascii=False)

    target_loop = MAIN_IOLOOP
    use_callback = (
        target_loop is not None
        and hasattr(target_loop, 'asyncio_loop')
        and target_loop.asyncio_loop.is_running()
    )

    for client in list(ws_clients):
        try:
            if use_callback:
                target_loop.add_callback(client.write_message, msg)
            else:
                client.write_message(msg)
        except Exception as e:
            print(f"[WS BROADCAST ERROR] {e}")


engine.register_word_callback(on_engine_word_recognized)


class LLMCombineHandler(tornado.web.RequestHandler):
    """API Ghép các từ rời rạc thành câu hoàn chỉnh bằng Google Gemini."""
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self):
        self.set_status(204)
        self.finish()

    async def post(self):
        try:
            req_data = json.loads(self.request.body.decode('utf-8'))
            words = req_data.get("words", [])
            style = req_data.get("style", "natural")
            custom_prompt = req_data.get("custom_prompt", "")
            api_key = req_data.get("api_key", "").strip()
            model = req_data.get("model", gemini_builder.model)

            if api_key:
                gemini_builder.set_api_key(api_key)

            # Chạy gọi API Gemini trong thread pool riêng để không bao giờ làm đơ Tornado event loop & WebSocket camera
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: gemini_builder.combine_words_to_sentence(
                    words=words,
                    style=style,
                    custom_prompt=custom_prompt,
                    model_override=model
                )
            )

            self.write(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({
                "success": False,
                "error": f"Lỗi xử lý server: {str(e)}"
            }, ensure_ascii=False))


class LLMTestKeyHandler(tornado.web.RequestHandler):
    """API Kiểm tra API Key của Google Gemini."""
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.set_header("Access-Control-Allow-Origin", "*")

    async def post(self):
        try:
            req_data = json.loads(self.request.body.decode('utf-8'))
            api_key = req_data.get("api_key", "").strip()
            model = req_data.get("model", gemini_builder.model)

            if model in SUPPORTED_MODELS:
                gemini_builder.set_model(model)

            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(
                None,
                lambda: gemini_builder.test_connection(api_key=api_key, model=model)
            )
            if res.get("valid"):
                gemini_builder.set_api_key(api_key)

            self.write(json.dumps(res, ensure_ascii=False))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"valid": False, "message": str(e)}, ensure_ascii=False))


class LabelsHandler(tornado.web.RequestHandler):
    """API trả về danh sách các từ/nhãn được hỗ trợ bởi 2 model."""
    def get(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        self.write(json.dumps({
            "one_hand": engine.one_hand_labels,
            "two_hands": engine.two_hands_labels
        }, ensure_ascii=False))


class CameraControlHandler(tornado.web.RequestHandler):
    """API điều khiển Camera Server."""
    async def post(self):
        try:
            req_data = json.loads(self.request.body.decode('utf-8'))
            action = req_data.get("action")
            device_id = int(req_data.get("device_id", 0))

            if action == "start":
                success = engine.start(device_id=device_id)
                self.write(json.dumps({"success": success, "status": engine.get_status()}))
            elif action == "stop":
                engine.stop()
                self.write(json.dumps({"success": True, "status": engine.get_status()}))
            elif action == "restart":
                engine.stop()
                await asyncio.sleep(0.5)
                success = engine.start(device_id=device_id)
                self.write(json.dumps({"success": success, "status": engine.get_status()}))
            else:
                self.write(json.dumps({"success": False, "message": "Unknown action"}))
        except Exception as e:
            self.write(json.dumps({"success": False, "error": str(e)}))


def periodic_broadcast():
    """Gửi cập nhật trạng thái định kỳ tới WebSocket."""
    RealtimeWebSocket.broadcast_status()


def make_app():
    static_path = os.path.join(os.path.dirname(__file__), "static")
    template_path = static_path

    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/video_feed", VideoFeedHandler),
        (r"/ws", RealtimeWebSocket),
        (r"/api/llm/combine", LLMCombineHandler),
        (r"/api/llm/test_key", LLMTestKeyHandler),
        (r"/api/labels", LabelsHandler),
        (r"/api/control/camera", CameraControlHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path}),
    ], template_path=template_path, static_path=static_path, debug=False)


def main():
    global MAIN_IOLOOP
    MAIN_IOLOOP = tornado.ioloop.IOLoop.current()

    parser = argparse.ArgumentParser(description="Sign Language Web App Server")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--autostart_server_cam", action="store_true", help="Tự động bật camera server OpenCV khi mở")
    args = parser.parse_args()

    engine.device_id = args.camera
    if args.autostart_server_cam:
        # Chỉ bật camera server khi có cờ yêu cầu rõ ràng, tránh xung đột thiết bị với Camera trình duyệt
        started = engine.start()
        if not started:
            print("[SERVER THONG BAO] Camera OpenCV phan cung chua san sang tren may chu.")
    else:
        print("[SERVER THONG BAO] Mặc định sử dụng CAMERA TRÌNH DUYỆT (HTML5 WebCam).")
        print("[SERVER THONG BAO] Nếu muốn dùng Camera Server OpenCV, bạn có thể chuyển tab trên giao diện web.")

    app = make_app()

    # Tự động tìm cổng khả dụng nếu cổng mặc định (8000) đang bị chiếm dụng
    port = args.port
    max_tries = 25
    bound = False
    server = None
    for p in range(port, port + max_tries):
        try:
            server = tornado.httpserver.HTTPServer(app)
            server.listen(p)
            if p != port:
                print(f"[SERVER THÔNG BÁO] Cổng {port} đang bận (có thể do một phiên làm việc khác đang chạy).")
                print(f"[SERVER THÔNG BÁO] Đã tự động chuyển sang Cổng khả dụng: {p}!")
            port = p
            bound = True
            break
        except OSError as e:
            if getattr(e, 'winerror', None) == 10048 or "10048" in str(e) or "already in use" in str(e).lower():
                continue
            raise

    if not bound:
        print(f"[LỖI] Không thể tìm thấy cổng mạng khả dụng từ {args.port} đến {args.port + max_tries}!")
        return

    # Đăng ký bộ đếm gửi trạng thái mỗi 500ms
    status_timer = tornado.ioloop.PeriodicCallback(periodic_broadcast, 500)
    status_timer.start()

    print("=" * 68)
    print(" 🚀 HỆ THỐNG WEB NHẬN DIỆN KÝ HIỆU & GHÉP CÂU GEMINI LLM ĐÃ SẴN SÀNG!")
    print(f" 🌐 Địa chỉ truy cập giao diện: http://localhost:{port}/")
    print(f" 🤖 Mô hình 1 Tay: {len(engine.one_hand_labels)} từ | Mô hình 2 Tay: {len(engine.two_hands_labels)} từ")
    print(f" 🧠 Gemini LLM Model: {DEFAULT_GEMINI_MODEL}")
    print("=" * 68)

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\n[SERVER] Đang dừng server...")
        status_timer.stop()
        engine.stop()
        tornado.ioloop.IOLoop.current().stop()
        print("[SERVER] Server đã đóng.")


if __name__ == "__main__":
    main()
