# Tập ground-truth cho OCR

Thước đo để trả lời một câu hỏi duy nhất: **thay đổi vừa rồi có làm OCR đọc đúng hơn không?**

Không có tập này thì mọi "cải thiện" chỉ là cảm giác.

## Vì sao cần

Bật dynamic tiling, đổi model, thêm tiền xử lý — mỗi thứ đều *có vẻ* đúng. Nhưng "có vẻ" không đủ để đốt 20 giờ GPU. Đo trước, quyết sau.

## Quy trình

### 1. Sinh mẫu (chạy trên Kaggle, sau khi có keyframe)

```bash
python ../build_ground_truth.py \
  --keyframes-parquet <đường dẫn>/keyframes.parquet \
  --media-root output/<release> \
  --out . --per-bucket 50
```

Sinh `sample.jsonl` — 150 dòng, chia 3 nhóm khó đều nhau, trường `text` để rỗng.

Cách phân nhóm: dựa trên độ nét (phương sai Laplacian) và độ tương phản. Đây chỉ là cách **trải mẫu cho đều**, không phải phán xét cuối cùng — mắt người mới quyết.

### 2. Gán nhãn tay

Mở từng ảnh, gõ **chính xác** chữ nhìn thấy vào trường `text`.

Quy ước:
- **Giữ nguyên dấu tiếng Việt.** Đây là điều quan trọng nhất — mất dấu là sai.
- Giữ nguyên chữ hoa/thường như trong ảnh (lúc đo sẽ tự hạ chữ thường)
- Nhiều dòng chữ → nối bằng khoảng trắng, theo thứ tự đọc tự nhiên (trên xuống, trái sang phải)
- Không có chữ nào → để `""` (chuỗi rỗng), **không** xóa dòng
- Chữ bị che một phần → chỉ gõ phần đọc được
- Không đoán chữ mờ không đọc nổi → bỏ qua phần đó

Xong đổi tên thành `labels.jsonl`.

**Nên gán 2 lượt cách nhau vài giờ**, đối chiếu chỗ khác nhau. Nhãn sai làm thước đo lệch, mà thước đo lệch còn tệ hơn không có thước đo.

### 3. Đo

```bash
# baseline trước khi sửa gì
python ../measure_ocr.py --labels labels.jsonl --out baseline-1b.json --label baseline-1b

# sau khi bật tiling
python ../measure_ocr.py --labels labels.jsonl --max-num 4 --out tiling-max4.json --label tiling-max4
```

## Đọc kết quả

CER = tỉ lệ ký tự sai. Thấp hơn là tốt hơn.

Nhìn **`by_difficulty` trước, CER tổng sau**. Lý do: cải thiện thường chỉ hiện ra ở nhóm `hard` (chữ nhỏ, mờ). CER tổng bị nhóm `easy` kéo xuống nên nhìn phẳng, dễ tưởng "không đổi gì".

Ngưỡng đã đặt trong kế hoạch: **CER nhóm `hard` giảm ≥15%**, CER tổng **không tăng**.

## Bẫy đã biết

**Unicode.** Chữ "ề" có hai cách mã hóa: một ký tự gộp, hoặc "e" cộng hai dấu rời. Nhìn giống hệt nhau trên màn hình nhưng máy coi là khác. Không quy về một dạng thì CER báo sai dù model đọc đúng hoàn toàn.

`system1.metrics` đã tự xử lý việc này (quy về NFC). Nếu tự viết mã đo ở chỗ khác, **nhớ chuẩn hóa trước**.

## File trong thư mục

| File | Nội dung |
|---|---|
| `sample.jsonl` | mẫu sinh tự động, nhãn còn rỗng |
| `labels.jsonl` | sau khi gán nhãn tay — đây là thước đo |
| `baseline-1b.json` | số gốc, trước mọi thay đổi |
| `tiling-max*.json` | kết quả từng mức `max_num` |
| `caption-comparison.json` | so caption Vintern-3B vs Qwen-7B |
| `preprocess-comparison.json` | so từng kỹ thuật OpenCV |
| `gate-distribution.json` | phân bố chỉ số cổng lọc |

`labels.jsonl` là tài sản của cả nhóm — gán một lần, dùng cho mọi so sánh về sau.
