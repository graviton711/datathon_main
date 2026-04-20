# Project Lessons & Behavioral Rules

## 1. EDA Workflow
- **Rule**: Khi thực hiện EDA, chỉ tạo file Notebook (`.ipynb`). 
- **Rule**: Notebook phải bao gồm logic tự động lưu ảnh vào thư mục `data/plots/`.
- **Constraint**: KHÔNG được tạo thêm các script Python riêng lẻ (`scripts/reproduce_*.py`) để chạy sinh ảnh. Dự án cần được giữ sạch sẽ.
- **Workflow**: Assistant viết code -> User chạy Notebook -> Assistant đọc ảnh từ `data/plots/` để phân tích.

## 2. Environment & Performance
- **PyArrow**: Luôn ưu tiên dùng engine `pyarrow` cho mọi thao tác đọc CSV/Parquet.
- **Dtype Management**: Luôn ép kiểu `str` cho các cột ID (`promo_id`, `order_id`, v.v.) để tránh lỗi DtypeWarning.

## 3. EDA Execution History (Milestones)
- **Phase 0 (Cleaning)**: Quy chuẩn hóa dữ liệu sang Parquet, ép kiểu ID, thống nhất đơn vị tiền tệ.
- **Phase 1 (Diagnostic)**: Phát hiện biến động Sentiment là "biến báo trước" (Leading Indicator) cho sự gãy đổ doanh số. Phân tích Category Streetwear là nòng cốt.
- **Phase 2 (Root Cause)**: Bác bỏ giả thuyết "đứt gãy chuỗi cung ứng" (Inventory ổn định). Xác định vấn đề nằm ở **Conversion collapse** do lỗi chọn size (wrong_size > 35%) và niềm tin khách hàng.
- **Phase 3 (Strategy)**: Xác nhận sự sụp đổ là **toàn diện toàn quốc** (Universal). Phát hiện chiến lược tăng giá sai thời điểm và thảm họa giữ chân khách hàng (Retention drop từ 65% xuống <10%).
- **Verification**: Đối soát 100% khớp dữ liệu giữa giao dịch gốc và báo cáo tổng.

## 4. Modeling Roadmap (Lessons Learned)
- **Regime Awareness**: Không được huấn luyện mô hình trên toàn bộ 10 năm một cách cào bằng. Cần trọng số cao cho dữ liệu sau 2019 (New Regime).
- **Critical Features**: Phải đưa biến `Sentiment`, `Size_Return_Rate` và `Price_Elasticity` vào Feature Engineering.
- **Data Safety**: Luôn đối soát tổng doanh thu sau khi ghép bảng (Reconciliation) để tránh mất dữ liệu khi train.
