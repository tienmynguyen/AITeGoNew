# 🤟 Standalone AI Sign Language Translator & Gemini LLM Sentence Builder

Ứng dụng Độc lập Nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL) phân nhánh thời gian thực kết hợp Trí tuệ Nhân tạo Google Gemini LLM để ghép câu tự nhiên.

Thư mục này được đóng gói **hoàn toàn độc lập (Standalone)**, chỉ chứa các **mô hình AI đã train**, mã nguồn suy luận (inference), và giao diện Desktop hoàn chỉnh. Không phụ thuộc vào dữ liệu thu thập thô hay các notebook huấn luyện.

---

## 📁 Cấu trúc Thư mục Độc lập

```text
standalone_app/
├── models/                         # Mô hình TFLite và danh sách nhãn đã train
│   ├── one_hand_classifier.tflite  # Model TFLite 1 tay (58 đặc trưng)
│   ├── two_hands_classifier.tflite # Model TFLite 2 tay (100 đặc trưng)
│   ├── labels_one_hand.csv         # 10 từ cử chỉ 1 tay
│   └── labels_two_hands.csv        # 28 từ cử chỉ 2 tay (bao gồm chữ số)
├── core/                           # Mã nguồn nhận diện & xử lý cốt lõi
│   ├── __init__.py
│   ├── sign_engine.py              # Động cơ nhận diện MediaPipe Holistic & Pipeline AI
│   ├── classifiers.py              # Trình nạp và suy luận TFLite đa luồng
│   ├── hand_router.py              # Bộ định tuyến 1 tay / 2 tay (Majority Voting)
│   ├── normalization.py            # Chuẩn hóa tọa độ theo Mid-Shoulder & Shoulder Width
│   ├── visualizer.py               # Vẽ HUD, nhãn tiếng Việt có dấu & thanh giữ 2s
│   ├── cvfpscalc.py                # Bộ tính FPS thời gian thực
│   └── gemini_service.py           # Dịch vụ ghép câu thông minh bằng Google Gemini LLM
├── static/                         # Giao diện người dùng hiện đại (Dark Theme)
│   ├── index.html                  # Giao diện chính tối giản, popup cài đặt
│   ├── css/
│   │   └── style.css               # Phong cách Glassmorphism, thanh tiến trình
│   └── js/
│       └── app.js                  # Xử lý WebSocket, WebCam, hàng đợi & TTS
├── app.py                          # File thực thi chính (Tornado Server + Desktop Window)
├── run_app.bat                     # File kích hoạt 1-click trên Windows
└── requirements.txt                # Danh sách thư viện cần thiết
```

---

## 🚀 Hướng dẫn Khởi chạy Nhanh

### Cách 1: Chạy bằng 1 cú nhấp chuột (Khuyến nghị trên Windows)
- Nhấp đúp chuột vào file: **`run_app.bat`**.
- Hệ thống sẽ tự động kích hoạt môi trường ảo Python và mở giao diện ứng dụng dưới dạng **Cửa sổ Desktop riêng biệt (Native App Window)**!

### Cách 2: Chạy bằng dòng lệnh Terminal
```bash
# Kích hoạt môi trường ảo (nếu có)
venv\Scripts\activate

# Di chuyển vào thư mục standalone_app và chạy
cd standalone_app
python app.py
```

---

## 🌟 Các Tính năng Nổi bật

1. **Hỗ trợ Camera Kép**:
   - **Camera Trình duyệt (HTML5 WebCam)**: Mặc định, độ trễ cực thấp, không chiếm quyền thiết bị hệ thống.
   - **Camera Server (OpenCV VideoCapture)**: Đọc trực tiếp phần cứng camera máy chủ với luồng MJPEG.
2. **Bộ định tuyến Bàn tay Thông minh (`HandRouter`)**:
   - Tự động nhận diện tư thế người dùng để kích hoạt nhánh mô hình 1 tay hoặc 2 tay phù hợp mà không cần thao tác thủ công.
3. **Cơ chế Giữ Cử Chỉ 2.0 Giây (Hold Duration 2s) & Chống Rung (Jitter Cushion)**:
   - Hiển thị thanh tiến trình đồ họa `0% -> 100% (2.0s)`.
   - Có dung sai 12 khung hình (~450ms) chống rung tay tự nhiên; khi giữ đủ 2.0s từ sẽ được chốt và tự động thêm vào hàng đợi.
4. **Tự Động Ghép Câu với Google Gemini LLM**:
   - Sau khi thêm hoặc xóa từ vựng trong hàng đợi, hệ thống tự động phân tích ngữ cảnh và ghép thành câu văn tiếng Việt tự nhiên, chuẩn ngữ pháp.
   - Hỗ trợ xử lý ghép số thông minh (VD: `giá` + `1` + `00` + `000` $\rightarrow$ `giá 100.000 VNĐ`).
   - Tự động chuyển đổi các model Gemini mới nhất (`gemini-3.8-flash`, `gemini-1.5-flash`, ...).
5. **Đọc Câu Hoàn Chỉnh (Text-to-Speech)**:
   - Nút phát âm thanh đọc câu hoàn chỉnh qua loa bằng giọng tiếng Việt tự nhiên.
6. **Từ điển 38 Từ Ký Hiệu & Popup Cài đặt**:
   - Màn hình chính tối giản chỉ gồm: Khung hình Camera, Hàng đợi từ và Câu hoàn chỉnh to rõ.
   - Toàn bộ từ điển, ngưỡng tin cậy, thời gian giữ cử chỉ, nhập Gemini API Key đều được gom gọn trong Popup Cài đặt.

