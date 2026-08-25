# Báo cáo Đánh giá Thử nghiệm ASR Tiếng Việt: 5 Cấu hình trên GPU

Báo cáo này so sánh hiệu năng, tốc độ, độ ổn định và chất lượng đầu ra của 5 cấu hình nhận dạng giọng nói tự động (ASR) tiếng Việt chạy trên **NVIDIA GeForce RTX 4060 Laptop GPU**:

1. **faster-whisper-medium** (CTranslate2 FP16)
2. **wav2vec 2.0** (`khanhld/wav2vec2-base-vietnamese-160h` FP32, raw CTC)
3. **faster-whisper-large-v3 (INT8)** (`large-v3` với CTranslate2 INT8 lượng tử hóa)
4. **Wav2Vec 2.0 + Punctuation Model** (`wav2vec2` kết hợp `dragonSwing/vibert-capu` phục hồi dấu câu & viết hoa)
5. **NVIDIA FastConformer** (`nvidia/parakeet-ctc-0.6b-vi` qua NeMo toolkit)

---

## 1. Kết Quả Đo Lường Hiệu Năng Thực Tế (GPU)

Thử nghiệm được thực hiện trên 2 tệp âm thanh WAV trích xuất từ video YouTube (tần số 16kHz, mono):
- **Tệp 1 (`1yHly8dYhIQ.wav`)**: Thời lượng **1266.60s** (~21.1 phút)
- **Tệp 2 (`fE0J-YeI5xc.wav`)**: Thời lượng **973.50s** (~16.2 phút)
- **Tổng Thời Lượng Âm Thanh**: **2240.10s** (~37.3 phút)

### Bảng thời gian xử lý (giây) và Hệ số thời gian thực (Real-Time Factor - RTF)

| Cấu hình ASR | Thời gian chạy Tệp 1 | Thời gian chạy Tệp 2 | Tổng thời gian chạy | Hệ số Tốc độ (x real-time) | Bộ nhớ VRAM khi chạy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **faster-whisper-medium (FP16)** | 90.07 s | 66.26 s | 156.33 s | **~14.3x** | ~1.5 GB |
| **wav2vec 2.0 (raw CTC)** | 6.35 s | 4.96 s | 11.31 s | **~198.1x** | ~380 MB |
| **faster-whisper-large-v3 (INT8)** | 39.74 s | 28.31 s | 68.05 s | **~32.9x** | ~1.6 GB |
| **Wav2Vec 2.0 + Punctuation** | 14.88 s | 11.65 s | 26.53 s | **~84.4x** | ~810 MB |
| **NVIDIA FastConformer** | 4.72 s | 3.64 s | 8.36 s | **~267.9x** | ~2.4 GB |

> [!NOTE]
> - **Real-Time Factor (RTF)** biểu thị số giây âm thanh được xử lý trong mỗi giây thực tế. Ví dụ, FastConformer chạy ở tốc độ ~267.9x tức là 1 giây chạy trên GPU sẽ nhận dạng được khoảng 4.4 phút âm thanh.
> - Thời gian chạy của cấu hình `Wav2Vec 2.0 + Punctuation` bao gồm thời gian chạy nhận diện của `wav2vec 2.0` và thời gian khôi phục dấu câu của model `vibert-capu` (giải mã văn bản chỉ mất ~3.91s cho cả 37 phút âm thanh).

---

## 2. Nhận Xét & So Sánh Chất Lượng Văn Bản Đầu Ara

### A. faster-whisper-medium (FP16) & large-v3 (INT8)
*   **Chất lượng văn bản**: Rất xuất sắc. Cả hai mô hình đều có cơ chế attention ngữ cảnh cao, tự động khôi phục dấu câu (`.`, `,`), viết hoa đầu câu, viết hoa tên riêng ("Thành phố Hồ Chí Minh", "Trung Bộ"), và chuẩn hóa định dạng số (`30 tháng 10`, `13 tuổi`).
*   **Điểm khác biệt của Large-v3 (INT8)**: 
    *   Nhờ **lượng tử hóa INT8 trên GPU**, mô hình `large-v3` (1.5B tham số) chạy nhanh gấp **2.3 lần** so với `medium` (769M tham số, FP16), trong khi độ chính xác và khả năng nhận diện từ khó cao hơn rõ rệt.
    *   VRAM tiêu thụ cực kỳ tiết kiệm (~1.6GB), tương đương bản medium FP16.

### B. wav2vec 2.0 (raw CTC)
*   **Chất lượng văn bản**: Văn bản hoàn toàn viết thường (lower-case), không có bất kỳ dấu câu nào, số viết hoàn toàn bằng chữ (`ba mươi mốt tháng mười`).
*   **Tốc độ**: Siêu nhanh (~198.1x), tuy nhiên do thiếu chuẩn hóa văn bản nên không thể dùng trực tiếp làm phụ đề mà không có hậu xử lý.

### C. Wav2Vec 2.0 + Punctuation Model (vibert-capu)
*   **Chất lượng văn bản**: Sau khi đi qua `vibert-capu`, toàn bộ văn bản thô của Wav2Vec2 đã được phục hồi dấu câu và viết hoa cực kỳ tự nhiên.
*   **Tốc độ**: Sự kết hợp này mang lại hiệu năng tối ưu nhất giữa tốc độ và chất lượng: Chạy nhanh gấp **6 lần** so với `faster-whisper-medium` và gần gấp **2.5 lần** so với `large-v3 (INT8)`.

### D. NVIDIA FastConformer (parakeet-ctc-0.6b-vi)
*   **Chất lượng văn bản**: FastConformer sử dụng bộ Tokenizer BPE được huấn luyện kèm dấu câu và viết hoa từ đầu. Do đó, văn bản đầu ra có sẵn dấu câu và viết hoa đầy đủ cực kỳ chính xác mà không cần mô hình hậu xử lý nào khác.
*   **Tốc độ**: Đứng đầu tuyệt đối về hiệu năng (**~267.9x**). Xử lý xong 21 phút âm thanh chỉ trong **4.72 giây**.

---

## 3. Tiêu Chí Đánh Giá Khi Mở Rộng Quy Mô Lớn (200,000 Samples)

Khi đưa hệ thống ASR vào môi trường production để xử lý tập dữ liệu cực lớn khoảng **200,000 mẫu âm thanh** (giả định độ dài trung bình mỗi mẫu là 10 giây, tương đương **555.6 giờ** âm thanh), chúng ta cần đánh giá dựa trên các tiêu chí cụ thể sau:

### A. Thời gian thực thi ước tính (Estimated Processing Time)
Thời gian cần thiết để hoàn thành xử lý toàn bộ 200,000 mẫu trên 1 GPU RTX 4060:
1.  **NVIDIA FastConformer**: **~2.07 giờ** (Vô địch tuyệt đối)
2.  **Wav2Vec 2.0 + Punctuation**: **~6.58 giờ**
3.  **faster-whisper-large-v3 (INT8)**: **~16.89 giờ**
4.  **faster-whisper-medium (FP16)**: **~38.85 giờ**

### B. Khả năng gộp nhóm xử lý (Batching & Throughput)
*   **Các mô hình CTC (FastConformer, Wav2Vec 2.0)**: Không tự hồi quy (Non-Autoregressive), thời gian xử lý tỷ lệ tuyến tính với độ dài âm thanh. Rất phù hợp với việc batching lớn (ví dụ: `batch_size=128` hoặc `256` trên GPU lớn). Tốc độ có thể tăng thêm 3-5 lần khi xử lý đồng thời nhiều mẫu nhỏ.
*   **Các mô hình Autoregressive (Whisper)**: Quá trình giải mã token xảy ra tuần tự từng bước, giới hạn khả năng song song hóa của GPU. Ở quy mô 200k mẫu, Whisper sẽ tạo ra hàng đợi xử lý rất dài.

### C. Độ ổn định và Rủi ro ảo giác (Hallucination & Stability)
*   **Whisper**: Dễ gặp lỗi lặp từ vô hạn (repetition loops) hoặc sinh văn bản ảo giác (hallucination) khi gặp các đoạn âm thanh im lặng dài hoặc tiếng ồn trắng. Điều này có thể làm treo hàng đợi xử lý hoặc sinh dữ liệu rác ở quy mô lớn.
*   **CTC Models (FastConformer, Wav2Vec 2.0)**: Không bao giờ bị lặp từ hoặc ảo giác trên các đoạn im lặng vì mô hình chỉ ánh xạ trực tiếp đặc trưng âm thanh sang ký tự. Cực kỳ bền bỉ và an toàn khi xử lý lô lớn không qua lọc nhiễu.

### D. Độ phức tạp của Pipeline & Bảo trì (Pipeline Maintenance & Cost)
*   **FastConformer / Whisper**: Single-stage pipeline. Chỉ cần tải một mô hình lên GPU, giảm thiểu độ trễ giao tiếp bộ nhớ và dễ dàng deploy lên các máy chủ chuyên dụng như NVIDIA Triton Inference Server.
*   **Wav2Vec2 + Punctuation**: Multi-stage pipeline. Yêu cầu nạp đồng thời 2 mô hình (ASR + BERT) lên GPU hoặc gọi API nối tiếp nhau. Điều này tăng độ phức tạp khi phân tải hệ thống (load balancing) và tăng nguy cơ nghẽn cổ chai (bottleneck) ở bước khôi phục dấu câu nếu số lượng luồng xử lý không đồng bộ tốt.

---

## 4. Khuyến Nghị Lựa Chọn Cho Quy Mô Lớn

1.  **Lựa chọn Tối ưu nhất (Về cả Tốc độ & Dấu câu)**: **NVIDIA FastConformer (`nvidia/parakeet-ctc-0.6b-vi`)**. Mô hình này có tốc độ xử lý nhanh nhất, có sẵn dấu câu/viết hoa tự nhiên và cực kỳ tiết kiệm thời gian khi chạy quy mô lớn.
2.  **Lựa chọn Khả thi thứ hai (Độ chính xác ngữ cảnh cao nhất)**: **faster-whisper-large-v3 (INT8)**. Nên sử dụng khi độ chính xác tuyệt đối là ưu tiên số một và âm thanh có nhiều tiếng ồn, từ chuyên ngành. Sử dụng cấu hình lượng tử hóa INT8 để tối đa hóa throughput trên GPU.

---

## Bổ sung: CER và hàm bàn giao (cập nhật 25/08/2026)

### Đo CER

Báo cáo trên tập trung vào tốc độ và nhận xét định tính. Đo CER/WER bằng công cụ dùng chung `system1.metrics` — có **chuẩn hoá Unicode NFC**, bắt buộc với tiếng Việt vì chữ có dấu tồn tại ở 2 dạng mã hoá.

```python
from system1.metrics import compute_cer, compute_wer
compute_cer(transcript_chuan, transcript_may_doc)
```

### Hàm bàn giao

```python
from system1.handoff import extract_audio_text

segments = extract_audio_text("clip.mp4")
# [{"start": 0.0, "end": 1.5, "text": "xin chào", "language": "vi"}, ...]
```

Trả về **kèm timestamp** như yêu cầu. File không có tiếng → danh sách rỗng, không báo lỗi. Đổi model bằng `provider="faster_whisper"`.

Hàm này bọc lại `system1.asr.transcribe_video` đã chạy trong pipeline — không viết lại logic, nên hành vi giống hệt bản sản xuất.

### Cấu hình sản xuất

| Mục | Giá trị |
|---|---|
| Model chính | `nvidia/parakeet-ctc-0.6b-vi` (NeMo FastConformer) |
| Dự phòng | `Systran/faster-whisper-large-v3` |
| Cắt đoạn | theo khoảng lặng ffmpeg, tối đa 12s |
| Tốc độ đo được | ~268× thời gian thực, ~2,4 GB VRAM |

ASR **không phải nút thắt** của pipeline: 4,7 giây cho 20 phút audio, so với ~40 phút cho OCR + caption cùng video. Tối ưu nên nhắm vào OCR/caption.
