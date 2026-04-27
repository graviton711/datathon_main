# DATATHON 2026: DỰ BÁO DOANH THU VÀ GIÁ VỐN HÀNG BÁN

## 1. Giới thiệu
Dự án này được thiết kế để dự báo doanh thu (Revenue) và giá vốn hàng bán (COGS) cho giai đoạn từ 01/01/2023 đến 01/07/2024 dựa trên dữ liệu lịch sử từ 2012-2022. Mô hình sử dụng phương pháp dự báo đệ quy (Recursive Forecasting) kết hợp với hiệu chuẩn động lực thị trường (Dynamic Momentum Calibration).

## 2. Cấu trúc thư mục

### Thư mục gốc
- data/: Lưu trữ dữ liệu thô và dữ liệu đã qua xử lý (parquet).
- docs/: Tài liệu hướng dẫn, lịch sử phiên làm việc và các thông tin thị trường đã xác minh.
- models/: Lưu trữ các tệp mô hình đã huấn luyện (.joblib).
- notebooks/: Các tập tin Jupyter Notebook phục vụ cho việc phân tích khám phá (EDA).
- src/: Mã nguồn chính của dự án.
- submissions/: Thư mục chứa các tệp kết quả cuối cùng (.csv).

### Chi tiết mã nguồn (src/)
- config.py: Cấu hình tập trung cho toàn bộ dự án, bao gồm các tham số mô hình, đường dẫn tệp và các hằng số thị trường.
- constants.py: Định nghĩa các hằng số cố định không thay đổi.

#### src/features/
- builder.py: Lớp BaselineFeatureExtractor thực hiện trích xuất đặc trưng thời gian, các sự kiện đặc biệt (Tết, payday) và tính toán tỷ trọng danh mục.

#### src/training/
- pipeline.py: Lớp ForecastingPipeline điều phối toàn bộ quy trình huấn luyện (fit) và dự báo (predict). Đây là điểm truy cập chính của hệ thống.
- analyst.py: Thực hiện phân tích momentum thị trường, tính toán quán tính (inertia) và hiệu chuẩn tăng trưởng YoY dựa trên dữ liệu lịch sử.
- weighting.py: Tính toán trọng số mẫu (sample weights) dựa trên độ mới của dữ liệu và các chỉ số bất thường để tối ưu hóa quá trình huấn luyện.

#### src/evaluation/
- evaluate.py: Công cụ đánh giá hiệu suất mô hình dựa trên các chỉ số MAE, RMSE và R2.
- benchmark_best.py: So sánh kết quả dự báo hiện tại với kết quả tốt nhất trên Leaderboard.

## 3. Hướng dẫn sử dụng

### Cài đặt môi trường
Yêu cầu Python 3.9+. Thực hiện cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Huấn luyện và Dự báo
Để thực hiện quy trình huấn luyện mô hình trên dữ liệu lịch sử và tạo tệp dự báo cho giai đoạn 2023-2024, chạy lệnh sau:
```bash
python src/training/pipeline.py
```
Kết quả sẽ được lưu tại `submissions/submission.csv`.

### Đánh giá mô hình
Để kiểm tra sai số của dự báo hiện tại so với các điểm chuẩn (benchmarks):
```bash
python src/evaluation/evaluate.py
```

## 4. Các quy tắc quan trọng
- Mọi thay đổi về tham số phải được thực hiện trong `src/config.py`.
- Tuyệt đối không sử dụng dữ liệu sau ngày 31/12/2022 trong quá trình huấn luyện để đảm bảo tính khách quan (Quy tắc 14).
- Các hệ số tăng trưởng phải được tính toán động từ dữ liệu huấn luyện, không sử dụng số liệu cố định (magic numbers).
