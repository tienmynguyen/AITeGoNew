# 🤟 AI Sign Language Recognition & Gemini LLM Sentence Builder

Ứng dụng Web Nhận diện Ngôn ngữ Ký hiệu Phân nhánh thời gian thực (Hierarchical Dynamic Sign Language Recognition) kết hợp Hệ thống Ghép câu Tự nhiên sử dụng **Google Gemini LLM**.

---

## 🌟 Tính Năng Nổi Bật

1. **Nhận diện Cử chỉ Tay Phân nhánh (Hierarchical Architecture)**:
   - **Tự động định tuyến (Hand Router)**: Tự nhận biết người dùng đang thực hiện cử chỉ 1 tay hay 2 tay bằng thuật toán Majority Voting.
   - **Nhánh 1 Tay (10 từ)**: Sử dụng mô hình `model/one_hand_classifier/one_hand_classifier.tflite` với 58 đặc trưng chuẩn hóa theo Mid-Shoulder.
   - **Nhánh 2 Tay (28 từ)**: Sử dụng mô hình `model/two_hands_classifier/two_hands_classifier.tflite` với 100 đặc trưng chuẩn hóa theo Mid-Shoulder kết hợp thân trên (MediaPipe Holistic).
   - Tùy chọn ép nhánh thủ công (Chỉ nhận 1 tay hoặc Chỉ nhận 2 tay).

2. **Hệ thống Ghép câu Thông minh với Gemini LLM**:
   - Sử dụng **Google Gemini API** (`gemini-3.8-flash`, `gemini-3.8-pro`, `gemini-2.5-flash`, `gemini-1.5-flash`) biến danh sách các từ rời rạc thành câu nói tiếng Việt hoàn chỉnh, chuẩn ngữ pháp giao tiếp (SVO).
   - Tự động gộp các cử chỉ bị nhận diện lặp lại khi giữ tay lâu.
   - Bổ sung hư từ, liên từ, trợ từ tình thái ("ạ", "dạ", "nhé", "thì", "là", "mà"...).
   - Hỗ trợ 5 phong cách câu:
     - **Tự nhiên**: Giao tiếp thường ngày.
     - **Lịch sự / Kính ngữ**: Dạ, thưa, ạ, tôn trọng đối phương.
     - **Ngắn gọn**: Súc tích, nhanh gọn.
     - **Câu hỏi / Nghi vấn**: Tự động chuyển đổi thành câu hỏi tự nhiên.
     - **Song ngữ (Việt - Anh)**: Dịch song ngữ phục vụ học tập hoặc giao tiếp quốc tế.

3. **Giao diện Web Trực quan & Đầy đủ Tiện ích**:
   - Camera trực tiếp (MJPEG Stream) với FPS cao và lớp phủ trực quan hóa (Live Prediction Overlay).
   - Điều chỉnh độ nhạy (Confidence Threshold) và thời gian trễ lặp (Debounce Cooldown) ngay trên giao diện.
   - Hàng đợi từ vựng trực quan (Word Chips) với chức năng xóa từng từ, hoàn tác từ cuối, thêm từ thủ công.
   - Từ điển 38 cử chỉ: Cho phép click vào bất kỳ từ nào để thêm nhanh vào hàng đợi để demo ngay cả khi không có camera.
   - **Phát âm Tiếng Việt (TTS)**: Đọc to câu nói đã được tạo bằng Web Speech Synthesis API.
   - Sao chép nhanh kết quả (Copy to clipboard) và lưu Lịch sử phiên dịch.

---

## 📁 Cấu Trúc Thư Mục `web_app/`

```text
web_app/
├── server.py              # Web Server (Tornado HTTP, MJPEG Stream, WebSocket & REST APIs)
├── sign_engine.py         # Động cơ nhận diện thị giác máy tính & TFLite 2 nhánh
├── gemini_service.py      # Tích hợp Gemini LLM Prompt Engineering & REST Client
├── run.bat                # Script chạy nhanh 1-click cho Windows
├── run.ps1                # Script chạy nhanh cho PowerShell
├── README.md              # Tài liệu hướng dẫn chi tiết
└── static/
    ├── index.html         # Giao diện chính (Dashboard)
    ├── css/
    │   └── style.css      # Giao diện Dark Cyber / Glassmorphism hiện đại
    └── js/
        └── app.js         # Xử lý sự kiện WebSocket, hàng đợi từ, gọi Gemini, TTS
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### Cách 1: Chạy bằng file Batch (Nhanh nhất trên Windows)
Nhấp đúp chuột vào file:
```
web_app\run.bat
```

### Cách 2: Chạy qua Terminal / PowerShell
```powershell
# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1

# Khởi chạy server web
python web_app/server.py --port 8000 --camera 0
```

Các tham số dòng lệnh tùy chọn:
- `--port <số_port>`: Đổi cổng mạng (mặc định: `8000`).
- `--camera <index>`: Chỉ định thiết bị camera (mặc định: `0`).
- `--no_cam_autostart`: Không tự động bật camera khi khởi động server.

---

## 🌐 Truy Cập Giao Diện
Mở trình duyệt web (Chrome, Edge, Firefox) và truy cập vào:
```
http://localhost:8000/
```

---

## 🔑 Cấu Hình Google Gemini API Key

1. Truy cập [Google AI Studio](https://aistudio.google.com/) và tạo một API Key miễn phí.
2. Trên giao diện Web app, nhấn nút **"⚙️ Cấu hình API"** ở góc phải thẻ Gemini.
3. Dán API Key vào ô `Gemini API Key`.
4. Nhấn **"Lưu Key"** và nhấn **"Kiểm tra"** để xác nhận kết nối thành công.

---

## 🎯 Ví Dụ Chuyển Đổi Ký Hiệu → Câu Hoàn Chỉnh

| Từ cử chỉ nhận diện | Phong cách | Câu hoàn chỉnh tạo bởi Gemini |
|---|---|---|
| `["xin chào", "tôi", "muốn", "giúp", "bạn"]` | Tự nhiên | *"Xin chào, tôi rất muốn giúp đỡ bạn!"* |
| `["cảm ơn", "bạn", "tốt"]` | Lịch sự | *"Dạ em xin cảm ơn anh/chị, anh/chị thật là tốt bụng ạ!"* |
| `["tên", "bạn", "cái gì"]` | Câu hỏi | *"Cho mình hỏi bạn tên là gì vậy?"* |
| `["giá", "bao nhiêu", "1", "00", "000"]` | Tự nhiên | *"Cái này giá bao nhiêu tiền vậy? Có phải 100.000 đồng không?"* |
| `["tạm biệt", "bạn", "yêu"]` | Song ngữ | *"Tạm biệt bạn, tôi luôn yêu quý bạn! \n(English: Goodbye my friend, I will always love you!)"* |

