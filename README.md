# Datathon 2026 — Deus x Machina
**VinUniversity | THE GRIDBREAKER: Breaking Business Boundaries**

> Dự báo doanh thu thương mại điện tử (01/01/2023 – 01/07/2024) cho doanh nghiệp thời trang Việt Nam.  

---

## Thành viên nhóm

| Tên | Email |
|-----|-------|
| Deus x Machina | dungroi19@gmail.com |

---

## Cấu trúc thư mục

```
├── data/
│   ├── raw/                  # Dữ liệu gốc từ ban tổ chức (14 file CSV)
│   └── processed/            # Dữ liệu Parquet sau tiền xử lý (Generated)
├── docs/                     # Quy chế và nhật ký dự án
├── notebooks/                # Notebook EDA và phân tích
├── reports/
│   ├── main.pdf              # Báo cáo chính thức
│   └── main.tex              # Source LaTeX
├── src/
│   ├── config.py
│   ├── constants.py
│   ├── utils/
│   │   └── prepare_data.py   # Script xử lý dữ liệu từ raw
│   ├── features/
│   │   └── builder.py
│   ├── training/
│   │   ├── pipeline.py
│   │   ├── analyst.py
│   │   └── weighting.py
│   └── evaluation/
│       └── evaluate.py
├── submissions/
└── requirements.txt
```

---

## Yêu cầu môi trường

- Python 3.12+
- Các thư viện chính: `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `lunardate`

Cài đặt:

```bash
# 1. Clone repository
git clone https://github.com/graviton711/datathon_main.git
cd datathon_main

# 2. Tạo và kích hoạt Virtual Environment
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Cài đặt dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Hướng dẫn tái lập kết quả

### Bước 1 — Chuẩn bị dữ liệu

1. Đặt toàn bộ 14 file CSV từ ban tổ chức vào thư mục `data/raw/`:

```
data/raw/
├── products.csv
├── customers.csv
├── promotions.csv
├── geography.csv
├── orders.csv
├── order_items.csv
├── payments.csv
├── shipments.csv
├── returns.csv
├── reviews.csv
├── sales.csv
├── sample_submission.csv
├── inventory.csv
└── web_traffic.csv
```

### Bước 2 — Tái lập toàn bộ kết quả

Sau khi đã có đủ 14 file CSV trong `data/raw/`, bạn chỉ cần chạy lệnh sau để thực hiện toàn bộ quy trình (Tiền xử lý -> Đánh giá CV -> Tạo Submission):

```bash
python reproduce.py
```

**Kết quả đầu ra:**
- **Logs:** Kết quả MAE trên các fold validation (2020-2022).
- **Submission:** `submissions/submission.csv` sẵn sàng nộp lên Kaggle.

---

## Tổng quan phương pháp

### Phân tích dữ liệu (EDA)

Bộ dữ liệu có một biến cố nổi bật: volume đơn hàng giảm ~40% vào năm 2019. Phân tích cho thấy đây là hệ quả của "Trust Collapse" — fill rate giữ ~100% nhưng Conversion Rate rơi từ 1.5% xuống 0.3%, và Year-2 retention từ 65% (2012) xuống còn 8% (2019). Nguyên nhân gốc rễ: 52.6% lý do trả hàng là `wrong_size` hoặc `not_as_described`.

### Mô hình dự báo

Pipeline sử dụng **Recursive LightGBM** kết hợp **Stationary Normalization**:

- **Stationary Normalization**: Chuẩn hoá target theo median năm `r̃_t = Revenue_t / μ̂_year` để xử lý đứt gãy cấu trúc 2019 (μ_2012–2018 ≈ 1.6× μ_2019+).
- **Regime Weighting**: Ưu tiên dữ liệu từ 2019 trở đi phản ánh chế độ thị trường hiện tại.
- **Blended Category Momentum**: Tính hệ số tăng trưởng theo cơ cấu danh mục sản phẩm.
- **Damped Multiplier**: Giảm chấn hệ số tăng trưởng theo độ sâu horizon để ngăn diverge qua 548 ngày.

| Cấu hình | MAE |
|----------|-----|
| Last Year Naive (baseline) | 859,676 |
| + LightGBM | 727,335 |
| + Stationary Norm | 684,330 |
| + Regime Weighting (LB) | **647,202** |

MAE 647,202 tương đương **14.7% MAPE** (baseline: 19.5%, cải thiện 4.8 pp).

---

## Kaggle

Submission: [https://drive.google.com/file/d/1UcSKK5Ngep4KYM2sKhKPqRbeht0r3y1u/view?usp=sharing](https://drive.google.com/file/d/1UcSKK5Ngep4KYM2sKhKPqRbeht0r3y1u/view?usp=sharing)
