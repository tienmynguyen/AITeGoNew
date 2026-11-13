# VSL 3D Studio - Vietnamese Sign Language 3D Avatar & Skeleton

Ứng dụng Web 3D hiển thị song song:
1. **Khung xương 3D Cyan (Cyan 3D Skeleton Overlay)** trực tiếp từ tọa độ `.npy`, bảo toàn 100% độ sâu không gian 3 chiều ($X, Y, Z$).
2. **Nhân vật 3D VRM 0.0 (`AvatarSample_A.vrm`)** chuyển động mượt mà thời gian thực, hỗ trợ cử chỉ đơn và ghép chuỗi đa từ (câu hoàn chỉnh) có nội suy cosine bridge mượt mà.

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh

### 1. Chạy Pipeline Chuyển Đổi Dữ Liệu `.npy` sang JSON
```bash
python scripts/convert_npy_to_json.py
```
*Tập lệnh quét dữ liệu `data/` ở chế độ CHỈ ĐỌC (Read-Only) và xuất ra thư mục `public/motions/` cùng `manifest.json`.*

### 2. Cài Đặt Gói Phụ Thuộc (Đã hoàn tất)
```bash
npm install
```

### 3. Khởi Chạy Giao Diện WebGL 3D Studio
```bash
npm run dev
```
Mở trình duyệt tại: **`http://localhost:5173`**

---

## 📁 Cấu Trúc Thư Mục Tự Chứa (Self-Contained inside `char/models/`)

```
char/models/
├── package.json               # Cấu hình phụ thuộc: three@0.149.0, @pixiv/three-vrm@1.0.9, kalidokit@1.1.5, vite
├── vite.config.js             # Cấu hình Vite bundler & alias
├── index.html                 # Giao diện Bento Studio UI chuẩn quốc tế
├── README.md                  # Hướng dẫn kỹ thuật chi tiết
├── scripts/
│   └── convert_npy_to_json.py # Pipeline Python chuyển đổi .npy sang 3D JSON
├── public/
│   ├── models/
│   │   └── AvatarSample_A.vrm # Mô hình nhân vật VRM 0.0 chuẩn
│   └── motions/
│       ├── manifest.json      # Danh mục từ vựng, số lượng frame và phân loại
│       ├── xin chào.json
│       ├── cảm ơn.json
│       ├── tôi.json
│       ├── bạn.json
│       └── ...                # 20 cử chỉ VSL (1 tay, 2 tay, chữ số 1-8)
└── src/
    ├── style.css              # Giao diện Bento Grid kính mờ (Glassmorphism dark theme)
    ├── main.js                # Quản lý cảnh Three.js, camera, lighting, render loop & UI events
    ├── skeletonOverlay.js     # Vẽ khung xương 3D Cyan (LineSegments + InstancedMesh khớp)
    ├── vrmEngine.js           # Bộ giải khớp VRM 0.0: Hierarchical arm alignment & finger curl solver
    ├── sequenceBlender.js     # Bộ ghép chuỗi đa từ, nội suy cosine 6 bridge frames & window.updateVSLSequence
    └── npyParser.js           # Bộ phân tích tệp nhị phân .npy kéo thả trực tiếp trên trình duyệt
```

---

## 🧠 Các Đặc Tính Kỹ Thuật Trọng Tâm

### 1. Bảo toàn độ sâu không gian 3D ($X, Y, Z$)
- Tọa độ Mid-Shoulder được neo chuẩn tại $(0, 1.26, 0.18)$ khớp với thân trên của nhân vật `AvatarSample_A`.
- Tỉ lệ đồng nhất $scale = 0.30$ trên cả 3 trục $X, Y, Z$ (không bị ép phẳng trục $Z$).
- Đối với dữ liệu 2D, pipeline tổng hợp độ sâu tự nhiên của MediaPipe: khuỷu tay gập về trước ($z = -0.12$), cổ tay đưa ra không gian trước ngực khi ký hiệu ($z = -0.25$), và 21 điểm bàn tay có vòm khum tự nhiên theo tỉ lệ co ngắn phân đoạn.

### 2. Bộ giải khớp VRM 0.0 (Self-Calibrating Bone Solver)
- Lưu trữ T-Pose chuẩn (`restLocalQuat`) của từng khớp ngay khi load mô hình.
- Cánh tay bên phải màn hình ($worldX > 0$) điều khiển `LeftUpperArm`, `LeftLowerArm`, `LeftHand` và các ngón tay trái.
- Cánh tay bên trái màn hình ($worldX < 0$) điều khiển `RightUpperArm`, `RightLowerArm`, `RightHand` và các ngón tay phải.
- Khớp cổ tay được kẹp góc xoay roll nhẹ trong phạm vi $\pm 65^\circ$, loại bỏ hoàn toàn hiện tượng lật ngược 180° lòng bàn tay.
- Ngón tay tính tỉ lệ duỗi `curlAmount` trong khoảng $[0.0, 1.0]$, cuộn tự nhiên vào lòng bàn tay theo chiều giải phẫu.

### 3. Ghép chuỗi câu liên tục (Multi-Word Chaining)
- Tự động chèn 6 khung hình cầu nối (Bridge Frames) có trọng số Cosine S-curve mượt mà giữa các từ:
  $$w(t) = \frac{1 - \cos(\pi \cdot t)}{2}$$
- Cắt bỏ 2 khung hình thừa ở ranh giới giữa các từ trong câu.
- Cung cấp hàm toàn cục `window.updateVSLSequence(['từ_1', 'từ_2', ...])` cho phép cập nhật hàng đợi câu ngay giữa lúc phát mà không bị giật khung hình.

