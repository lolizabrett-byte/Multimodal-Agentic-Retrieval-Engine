# Bảng Nhận Xét & So Sánh Các Pipeline OCR trên GPU

Dưới đây là bảng đánh giá chi tiết về tốc độ (Latency), độ chính xác nhận diện tiếng Việt (CER/WER) và khả năng ứng dụng thực tế của từng mô hình sau khi chạy thực nghiệm trên GPU **NVIDIA GeForce RTX 4060 Laptop**.

---

## 1. Bảng So Sánh Tổng Quan

| Pipeline | Thời gian trung bình (s) | Chất lượng Tiếng Việt | Ưu điểm | Nhược điểm | Đánh giá chung |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vintern-1B-v3_5 (Mới)** | 0.6933s | **Xuất sắc (9.8/10)** | - **Độ chính xác Word Error Rate (WER) cao nhất (0.34)**.<br>- Hiểu ngữ cảnh câu, giữ dấu câu chuẩn xác.<br>- Xử lý đa dòng và chữ nghiêng/mờ cực tốt. | - Thời gian xử lý chậm hơn các mô hình truyền thống (0.69s). | **Khuyên dùng nhiều nhất** cho các tác vụ cần độ chính xác cao về ngữ nghĩa và đọc hiểu văn bản phức tạp. |
| **EasyOCR** | 0.0605s | **Rất tốt (8.5/10)** | - Nhận diện dấu Tiếng Việt rất chuẩn.<br>- Tốc độ rất nhanh trên GPU (0.06s). | - Bị lỗi nếu hộp chữ quá nhỏ hoặc mờ. | **Khuyên dùng** cho ứng dụng thực tế cần cân bằng giữa tốc độ và chất lượng tiếng Việt cơ bản. |
| **PaddleOCR Only** | **0.0386s** | Khá (7.0/10) | - Tốc độ nhanh nhất.<br>- Nhận diện biên chữ (detector) cực kỳ chính xác. | - Hay làm mất hoặc sai dấu tiếng Việt do thiếu bộ từ điển chuẩn. | **Phù hợp nhất** khi cần xử lý thời gian thực (Real-time) và không quá khắt khe về chính tả dấu câu. |
| **PaddleOCR + VietOCR** | 0.1222s | Tệ (2.0/10) | - Phân vùng chữ tốt bằng Paddle. | - Phiên bản VietOCR (`vgg_seq2seq`) bị lỗi lặp chuỗi vô nghĩa (`001000000100`). | **Cần nâng cấp** lên VietOCR ResNet-Transformer hoặc thay thế hoàn toàn. |
| **PaddleOCR + TrOCR** | 0.1781s | Rất tệ (1.5/10) | - Nhận diện chữ tiếng Anh tốt. | - Model gốc `trocr-base-printed` chỉ hỗ trợ Tiếng Anh nên không thể nhận diện Tiếng Việt. | **Không khuyên dùng** cho dữ liệu tiếng Việt trừ phi fine-tune lại TrOCR. |
| **Qwen2-VL-2B-Instruct (Mới)** | 1.2546s | Tệ (3.0/10) | - Nhận diện đa ngôn ngữ tốt. | - Gặp lỗi tự chối phản hồi (refusal responses) bằng tiếng Việt.<br>- Độ trễ khá lớn (1.25s). | **Không khuyên dùng** cho tác vụ trích xuất văn bản tiếng Việt thuần túy. |
| **Florence-2** | 2.5993s | Rất tệ (1.0/10) | - Xuất ra cấu trúc JSON hoặc Bounding Box tốt. | - Tốc độ rất chậm.<br>- Bị lỗi ảo giác (hallucination) lặp từ vô hạn khi gặp Tiếng Việt. | **Không phù hợp** cho bài toán OCR Tiếng Việt. |

---

## 2. Nhận Xét Chi Tiết & Khuyến Nghị

### 🥇 Top 1 Về Chất Lượng: Vintern-1B-v3_5 (SOTA VLM)
- **Độ chính xác Tiếng Việt:** Vượt trội hoàn toàn so với các OCR truyền thống nhờ kiến trúc Vision-Language Model. Tỉ lệ lỗi từ (WER) chỉ **0.34**, giảm **~43%** số từ bị sai so với PaddleOCR. Vintern nhận diện chính xác các đại lượng số, ký tự đặc biệt, viết hoa, và xuống dòng tự nhiên mà không bị mất ngữ nghĩa.
- **Tốc độ:** Với trung bình **0.6933 giây/ảnh** trên GPU RTX 4060, Vintern cực kỳ lý tưởng để xử lý các tài liệu quét, keyframe chứa văn bản dài hoặc biển hiệu quảng cáo phức tạp.

### ⚡ Top 1 Về Tốc Độ: PaddleOCR Only
- **Tốc độ:** Đạt hiệu năng ấn tượng **0.0386 giây/ảnh**, nhanh gấp 18 lần so với VLM.
- **Ứng dụng:** Thích hợp nhất cho các bài toán nhận diện thời gian thực (Real-time) từ luồng Camera/Video trực tiếp.

### ⚠️ Lưu ý về các mô hình lớn đa ngôn ngữ (Qwen2-VL, Florence-2)
- Các dòng VLM đa ngôn ngữ lớn như **Qwen2-VL** hoặc **Florence-2** không được tối ưu hóa riêng cho bộ ngữ âm tiếng Việt, dẫn đến hiện tượng từ chối trích xuất (refusal) hoặc rơi vào vòng lặp vô hạn (hallucination). Không nên áp dụng trực tiếp các mô hình này trong môi trường sản phẩm (production) trừ khi có sự huấn luyện bổ sung (fine-tuning).


---

## 3. Bổ sung: CER và tiền xử lý (cập nhật 25/08/2026)

Bảng trên đo WER. Bản phân công yêu cầu **cả CER** — bổ sung bằng công cụ đo dùng chung.

### Cách đo

```bash
# baseline
python ../measure_ocr.py --labels ../ground_truth/labels.jsonl \
  --out ../ground_truth/baseline-1b.json --label baseline-1b

# so từng kỹ thuật tiền xử lý
python ../preprocess_experiment.py --labels ../ground_truth/labels.jsonl \
  --out ../ground_truth/preprocess-comparison.json
```

Thước đo dùng `system1.metrics`, **chuẩn hoá Unicode NFC** trước khi so. Không chuẩn hoá thì chữ "ề" mã hoá 2 kiểu bị tính là sai dù model đọc đúng.

### Tiền xử lý OpenCV — cảnh báo trước khi áp dụng

Threshold / morphology / grayscale sinh ra cho OCR đời cũ (Tesseract, PaddleOCR). **Vintern là VLM học trên ảnh màu tự nhiên** — ép ảnh về nhị phân đưa nó về dạng chưa từng thấy khi huấn luyện.

Đo trực tiếp trên một keyframe thật (640×360):

| Kỹ thuật | Số màu còn lại | Nhóm |
|---|---|---|
| gốc | ~58.000 | — |
| CLAHE | 58.580 | an toàn |
| khử nhiễu | 25.474 | an toàn |
| làm nét | 91.015 | an toàn |
| phóng to ×2 | 211.104 | an toàn |
| chuyển xám | 240 | rủi ro |
| **nhị phân (Otsu)** | **2** | **rủi ro** |
| **nhị phân thích ứng** | **2** | **rủi ro** |
| morphology | 236 | rủi ro |

Nhị phân hoá bỏ đi 99,997% thông tin màu. Với OCR đời cũ đó là dọn nhiễu; với VLM đó là mất dữ liệu.

**Quy tắc:** chỉ giữ kỹ thuật nào làm CER **giảm thật** trên tập nhãn. Kỹ thuật làm CER tăng thì ghi lại kèm số và không dùng — đó là kết quả có giá trị, không phải thất bại.

### Cấu hình dùng cho sản xuất

| Mục | Giá trị | Lý do |
|---|---|---|
| Model | `5CD-AI/Vintern-1B-v3_5` | thắng bake-off (WER 0,34) |
| dtype | `float16` | checkpoint là fp32; fp16 giảm nửa VRAM (3,75 → 1,88 GB) |
| `device_map` | `cuda:0` | `auto` xẻ model dù nó vừa 1 card |
| `max_dynamic_patch` | 4 | trần của model; keyframe 16:9 thành 2 patch |
| Tiền xử lý | chưa bật | chờ số đo từ `preprocess-comparison.json` |
