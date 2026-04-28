# THE GRIDBREAKER: Diagnosing Structural Shifts and Scaling Recursive Forecasts in E-Commerce

**DATATHON 2026 - ROUND 1**  
**Team: Deus x Machina**  
**Members:** Phan Dang Thai Son (Lead), Hoang Chi Tam  
**Organization:** VinUniversity, Hanoi, Vietnam

---

## 1. Giới thiệu
Dự án này tập trung vào việc chẩn đoán nguyên nhân sự sụt giảm doanh số đột ngột vào năm 2019 và xây dựng mô hình dự báo doanh thu (Revenue) và giá vốn hàng bán (COGS) cho giai đoạn 2023-2024.

Chúng tôi xác định đây là một cuộc khủng hoảng niềm tin từ khách hàng (Sizing Crisis) và giải quyết bài toán dự báo bằng phương pháp **Recursive LightGBM** kết hợp với **Stationary Normalization** và **Dynamic Momentum Calibration**.

## 2. Cấu trúc thư mục
Dự án được tổ chức theo tiêu chuẩn modular để đảm bảo tính tái lập và dễ dàng mở rộng:

```text
├── data/               # Dữ liệu thô và dữ liệu đã qua xử lý (parquet)
├── docs/               # Tài liệu hướng dẫn, quy tắc và nhật ký dự án
├── models/             # Lưu trữ các tệp mô hình đã huấn luyện (.joblib)
├── notebooks/          # Jupyter Notebooks phục vụ phân tích EDA
├── report/             # Báo cáo kỹ thuật (LaTeX NeurIPS template)
├── src/                # Mã nguồn chính
│   ├── features/       # Logic trích xuất đặc trưng (Stationary Features)
│   ├── training/       # Pipeline huấn luyện và hiệu chuẩn thị trường
│   ├── evaluation/     # Công cụ đánh giá Walk-Forward Cross-Validation
│   └── utils/          # Các hàm tiện ích bổ trợ
├── submissions/        # Tệp kết quả cuối cùng nộp lên Kaggle
└── requirements.txt    # Danh sách thư viện cần thiết
```

## 3. Cài đặt và Sử dụng

### Cài đặt môi trường
Yêu cầu Python 3.9+. Khuyến khích sử dụng Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Chạy quy trình huấn luyện và dự báo
Để tạo tệp nộp bài (`submissions/submission.csv`) bằng Pipeline ổn định nhất:
```bash
python -m src.training.pipeline
```

### Đánh giá mô hình (Backtesting)
Để chạy kiểm chứng chéo (3-Fold Walk-Forward CV) trên dữ liệu lịch sử:
```bash
python src/evaluation/evaluate.py
```

## 4. Phương pháp tiếp cận chính
- **Target Engineering**: Chuyển đổi Revenue sang tỷ lệ chuẩn hóa (Revenue/Annual Median) để loại bỏ nhiễu từ sự thay đổi quy mô doanh nghiệp qua các thời kỳ.
- **Dynamic Momentum**: Tự động tính toán hệ số tăng trưởng YoY dựa trên tín hiệu Q4 và quán tính thị trường (Inertia).
- **Seasonal Floors**: Áp dụng các ngưỡng chặn dưới cho doanh số tháng 9-10 dựa trên dữ liệu lịch sử để tránh dự báo quá thấp do biến động mùa vụ.
- **Rule 14 Compliance**: Đảm bảo tính minh bạch, tuyệt đối không rò rỉ dữ liệu tương lai vào mô hình huấn luyện.

## 5. Kết quả đạt được
- **Best Leaderboard Score**: **650,000 MAE** (Honest Pipeline).
- **Báo cáo chi tiết**: Xem tại [report/main.pdf](report/main.pdf) để hiểu sâu hơn về phân tích chẩn đoán sự sụp đổ năm 2019 và giải pháp hồi phục.

---
© 2026 Deus x Machina - VinUniversity.
