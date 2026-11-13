# Hierarchical Router Sign Language Recognition System

Hệ thống Nhận diện Ngôn ngữ Ký hiệu Động với **Kiến trúc Mô hình Phân nhánh (Hierarchical Router Architecture)** gồm 2 nhánh độc lập: **Nhánh 1 Tay (One-Hand Branch)** và **Nhánh 2 Tay (Two-Hands Branch)** kết hợp với Bộ định tuyến (Hand Router) tự động.

---

## 🌟 Kiến trúc Hệ thống Phân nhánh

```
                                    ┌────────────────────────┐
                                    │   Camera Input Frame   │
                                    └───────────┬────────────┘
                                                │
                                    ┌───────────▼────────────┐
                                    │   MediaPipe Holistic   │
                                    └───────────┬────────────┘
                                                │
                                    ┌───────────▼────────────┐
                                    │    router/hand_router  │
                                    │   (Automatic Router)   │
                                    └─────┬────────────┬─────┘
                                          │            │
                         ONE_HAND Mode    │            │    TWO_HANDS Mode
                   ┌──────────────────────┘            └──────────────────────┐
                   │                                                          │
     ┌─────────────▼─────────────┐                              ┌─────────────▼─────────────┐
     │ utils/normalization       │                              │ utils/normalization       │
     │ normalize_one_hand (58D)  │                              │ normalize_two_hands (100D)│
     └─────────────┬─────────────┘                              └─────────────┬─────────────┘
                   │                                                          │
     ┌─────────────▼─────────────┐                              ┌─────────────▼─────────────┐
     │ model/one_hand_classifier │                              │ model/two_hands_classifier│
     │ one_hand_classifier.tflite│                              │two_hands_classifier.tflite│
     └─────────────┬─────────────┘                              └─────────────┬─────────────┘
                   │                                                          │
                   └──────────────────────┬───────────────────────────────────┘
                                          │
                               ┌──────────▼───────────┐
                               │   utils/visualizer   │
                               │  (Prediction Badge)  │
                               └──────────────────────┘
```

### 1. Bộ định tuyến Tự động (Router):
- `router/hand_router.py`: Phân tích kết quả từ MediaPipe thu thập số lượng bàn tay xuất hiện và áp dụng bầu chọn đa số (Majority Voting) làm mịn.
- Tự động điều hướng luồng dữ liệu sang **Nhánh 1 Tay (`ONE_HAND`)** hoặc **Nhánh 2 Tay (`TWO_HANDS`)**.

### 2. Nhánh 1 Tay (One-Hand Branch):
- **Chuẩn hóa Tọa độ**: Gốc tương đối $(0, 0)$ đặt tại Trung điểm hai vai (Mid-Shoulder), chia tỉ lệ theo Khoảng cách hai vai (Shoulder Width). Vector đặc trưng 58 chiều (Thân trên + Bàn tay hoạt động).
- **Mô hình**: `model/one_hand_classifier/one_hand_classifier.tflite`
- **Dữ liệu & Nhãn**: `data/one_hand/` (Ví dụ nhãn: "Like", "OK", "Peace", "Point", "Stop").

### 3. Nhánh 2 Tay (Two-Hands Branch):
- **Chuẩn hóa Tọa độ**: Gốc tương đối $(0, 0)$ đặt tại Trung điểm hai vai (Mid-Shoulder), chia tỉ lệ theo Khoảng cách hai vai (Shoulder Width). Vector đặc trưng 100 chiều (Thân trên + 2 Tay).
- **Mô hình**: `model/two_hands_classifier/two_hands_classifier.tflite`
- **Dữ liệu & Nhãn**: `data/two_hands/` (Ví dụ nhãn: "XinChao", "CamOn", "XinLoi", "YeuThuong", "TamBiet").

---

## 📁 Cấu trúc Thư mục Dự án

```text
├── app.py                         # Ứng dụng chính (Chạy webcam thời gian thực & Thu thập dữ liệu)
├── router/
│   └── hand_router.py             # Bộ định tuyến xác định chế độ 1 tay hay 2 tay
├── utils/
│   ├── normalization.py           # Thuật toán chuẩn hóa tọa độ cho cả 1 tay và 2 tay
│   ├── cvfpscalc.py               # Đo FPS
│   └── visualizer.py              # Vẽ UI, bounding box, nhãn và trạng thái nhánh lên frame
├── data/
│   ├── one_hand/                  # Dữ liệu & nhãn từ cử chỉ 1 tay
│   │   ├── keypoint_classifier_label.csv
│   │   └── {label_name}/*.npy
│   └── two_hands/                 # Dữ liệu & nhãn từ cử chỉ 2 tay
│       ├── keypoint_classifier_label.csv
│       └── {label_name}/*.npy
├── model/
│   ├── one_hand_classifier/       # TFLite model và wrapper class cho 1 tay
│   │   ├── one_hand_classifier.py
│   │   └── one_hand_classifier.tflite
│   └── two_hands_classifier/      # TFLite model và wrapper class cho 2 tay
│       ├── two_hands_classifier.py
│       └── two_hands_classifier.tflite
└── notebooks/
    ├── train_one_hand.ipynb       # Notebook huấn luyện mô hình 1 tay
    └── train_two_hands.ipynb      # Notebook huấn luyện mô hình 2 tay
```

---

## 🚀 Hướng dẫn Sử dụng

### 1. Khởi chạy Ứng dụng OpenCV Thời gian thực
```bash
.\venv\Scripts\Activate.ps1
python app.py
```
- Phím **`m`** hoặc **`Tab`**: Chuyển chế độ Suy luận <-> Thu thập Dữ liệu.
- Phím **`b`**: Đổi nhanh nhánh qua lại giữa 1 Tay và 2 Tay (hoạt động ở cả 2 chế độ).
- **Ở Chế độ Suy luận (Mode 0)**:
  - Phím **`1`**: Ép chọn nhánh 1 tay (`ONE_HAND`).
  - Phím **`2`**: Ép chọn nhánh 2 tay (`TWO_HANDS`).
  - Phím **`3`** hoặc **`a`**: Chế độ Tự động (`Auto Router`).
- **Ở Chế độ Thu thập Dữ liệu (Mode 1)**:
  - Phím **`0-5`**: Bắt đầu thu thập **40 mẫu liên tục** (mặc định 40 mẫu/đợt, nghỉ 1s giữa mỗi mẫu, tùy chỉnh bằng `--samples` và `--rest_sec`).
  - Phím **`+`** / **`-`**: Tăng hoặc giảm nhanh số lượng mẫu cần thu thập trong mỗi đợt (10, 20, 30, 40, 50, ...).
  - Phím **`n`** / **`p`**: Lật trang danh sách từ (khi danh sách có nhiều hơn 6 từ).
  - Phím **`b`**: Chuyển nhánh thu thập (`data/one_hand/` <-> `data/two_hands/`).
  - Phím **`c`**: Hủy/dừng đợt thu thập đang chạy.
- Phím **`q`** hoặc **`ESC`**: Thoát chương trình.

### 2. Huấn luyện các Nhánh Mô hình
- Mở `notebooks/train_one_hand.ipynb` để huấn luyện & xuất file `model/one_hand_classifier/one_hand_classifier.tflite`.
- Mở `notebooks/train_two_hands.ipynb` để huấn luyện & xuất file `model/two_hands_classifier/two_hands_classifier.tflite`.
