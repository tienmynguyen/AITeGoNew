#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phần mềm Nhận diện Ký hiệu & Ghép câu Tự nhiên Google Gemini LLM (Standalone Desktop App).
Chạy độc lập 100% với:
  - 2 Mô hình TFLite (1 Tay & 2 Tay)
  - Bộ định tuyến bàn tay thông minh HandRouter
  - Hàng đợi từ vựng & Cơ chế giữ cử chỉ 2.0s chống rung
  - Tự động ghép câu qua Gemini LLM
  - Giao diện Desktop hiện đại, trực quan (Native App Window)
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import argparse
import threading
import subprocess
import webbrowser

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'matplotlib_vsl')

APP_DIR = os.path.abspath(os.path.dirname(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.httpserver
import tornado.iostream
import cv2 as cv
import numpy as np

from core.sign_engine import SignRecognitionEngine
from core.gemini_service import GeminiSentenceBuilder, SUPPORTED_MODELS, DEFAULT_GEMINI_MODEL


# Khởi tạo Singletons
engine = SignRecognitionEngine()
gemini_builder = GeminiSentenceBuilder()

# Quản lý kết nối WebSocket
ws_clients = set()
MAIN_IOLOOP = None


class MainHandler(tornado.web.RequestHandler):
    """Trang chủ ứng dụng."""
    def get(self):
        self.render("index.html")


class VideoFeedHandler(tornado.web.RequestHandler):
    """Luồng MJPEG Video Stream thời gian thực từ Camera Server (OpenCV)."""
    async def get(self):
        self.set_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.set_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.set_header('Pragma', 'no-cache')
        self.set_header('Expires', '0')

        blank_frame = None
        try:
            while True:
                conn = self.request.connection
                if hasattr(conn, 'stream') and conn.stream and conn.stream.closed():
                    break

                jpeg_bytes = engine.get_latest_jpeg()
                if jpeg_bytes is None:
                    if blank_frame is None:
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
    """WebSocket thời gian thực kết nối Camera Trình duyệt & điều khiển hệ thống."""
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
        # 1. Nhận frame nhị phân từ WebCam
        if isinstance(message, bytes):
            try:
                nparr = np.frombuffer(message, np.uint8)
                frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
                if frame is not None:
                    # Lật gương khung hình chuẩn góc nhìn selfie (đồng nhất với Server Camera)
                    frame = cv.flip(frame, 1)
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


def on_engine_word_recognized(event: dict):
    """Callback chuyển tiếp sự kiện từ mới tới tất cả các WebSocket client (an toàn đa luồng)."""
    word = event.get("word")
    branch = event.get("branch")
    conf = event.get("confidence")
    print(f"[EVENT] >>> CHỐT TỪ: '{word}' ({branch}, conf={conf}) -> Đang gửi tới {len(ws_clients)} WebSocket client(s)...")
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
    """API trả về danh sách các từ/nhãn được hỗ trợ."""
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
    RealtimeWebSocket.broadcast_status()


def make_app():
    static_path = os.path.join(APP_DIR, "static")
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


def launch_native_window(url: str, title: str = "AI Sign Language Translator"):
    """
    Khởi tạo cửa sổ Desktop độc lập:
      - Ưu tiên 1: pywebview nếu có
      - Ưu tiên 2: Microsoft Edge App Mode (--app=url) tạo cửa sổ native không thanh URL
      - Ưu tiên 3: Google Chrome App Mode
      - Ưu tiên 4: Mở trình duyệt mặc định
    """
    time.sleep(1.0)  # Chờ server lắng nghe

    # 1. Thử pywebview nếu được cài đặt
    try:
        import webview
        webview.create_window(title, url, width=1280, height=840, resizable=True)
        webview.start()
        return
    except ImportError:
        pass

    # 2. Tìm Microsoft Edge App Mode trên Windows
    edge_paths = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in edge_paths:
        if os.path.exists(p):
            try:
                subprocess.Popen([p, f"--app={url}", "--window-size=1280,840"])
                print("[DESKTOP APP] Đã mở cửa sổ ứng dụng Desktop (Microsoft Edge App Mode).")
                return
            except Exception:
                pass

    # 3. Tìm Google Chrome App Mode
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for p in chrome_paths:
        if os.path.exists(p):
            try:
                subprocess.Popen([p, f"--app={url}", "--window-size=1280,840"])
                print("[DESKTOP APP] Đã mở cửa sổ ứng dụng Desktop (Google Chrome App Mode).")
                return
            except Exception:
                pass

    # 4. Fallback: Mở trình duyệt mặc định
    print("[DESKTOP APP] Đang mở trình duyệt mặc định...")
    webbrowser.open(url)


def main():
    global MAIN_IOLOOP
    MAIN_IOLOOP = tornado.ioloop.IOLoop.current()

    parser = argparse.ArgumentParser(description="Standalone AI Sign Language Translator App")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--autostart_server_cam", action="store_true", help="Tự động bật Camera Server OpenCV")
    parser.add_argument("--no-window", action="store_true", help="Không tự mở cửa sổ ứng dụng Desktop")
    args = parser.parse_args()

    engine.device_id = args.camera
    if args.autostart_server_cam:
        started = engine.start()
        if not started:
            print("[THÔNG BÁO] Camera OpenCV phần cứng chưa sẵn sàng.")
    else:
        print("[THÔNG BÁO] Chế độ mặc định: Camera Trình duyệt (HTML5 WebCam).")

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
                print(f"[THÔNG BÁO] Cổng {port} đang bận (có thể do phiên làm việc trước hoặc web_app đang chạy).")
                print(f"[THÔNG BÁO] Hệ thống tự động chuyển sang Cổng khả dụng: {p}!")
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

    url = f"http://127.0.0.1:{port}/"
    print("=" * 70)
    print(" 🤟 PHẦN MỀM NHẬN DIỆN KÝ HIỆU & GHÉP CÂU GEMINI LLM ĐỘC LẬP")
    print(f" 🌐 Địa chỉ ứng dụng: {url}")
    print(f" 🤖 AI Model 1 Tay: {len(engine.one_hand_labels)} từ | Model 2 Tay: {len(engine.two_hands_labels)} từ")
    print(f" 🧠 Gemini LLM: {DEFAULT_GEMINI_MODEL}")
    print("=" * 70)

    # Mở cửa sổ Desktop nếu không tắt cờ --no-window
    if not args.no_window:
        t = threading.Thread(target=launch_native_window, args=(url,), daemon=True)
        t.start()

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\n[APP] Đang dừng phần mềm...")
        status_timer.stop()
        engine.stop()
        tornado.ioloop.IOLoop.current().stop()
        print("[APP] Phần mềm đã đóng hoàn tất.")


if __name__ == "__main__":
    main()

