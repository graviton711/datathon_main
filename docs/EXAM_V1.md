# ĐỀ THI VÒNG 1 — DATATHON 2026
## THE GRIDBREAKER: Breaking Business Boundaries
**Hosted by VinTelligence — VinUniversity Data Science & AI Club**

*Cuộc thi Khoa học Dữ liệu đầu tiên tại VinUniversity*
*Biến Dữ liệu thành Giải pháp cho Doanh nghiệp*

---

## Mục lục
1. [Mô tả Dữ liệu](#1-mô-tả-dữ-liệu)
2. [Đề Bài](#2-đề-bài)
    - [Phần 1: Câu hỏi Trắc nghiệm](#phần-1-câu-hỏi-trắc-nghiệm)
    - [Phần 2: Trực quan hoá và Phân tích Dữ liệu](#phần-2-trực-quan-hoá-và-phân-tích-dữ-liệu)
    - [Phần 3: Mô hình Dự báo Doanh thu (Sales Forecasting)](#phần-3-mô-hình-dự-báo-doanh-thu-sales-forecasting)
3. [Thang điểm Chấm thi](#3-thang-điểm-chấm-thi)
4. [Hướng dẫn Nộp bài](#4-hướng-dẫn-nộp-bài)

---

## 1. Mô tả Dữ liệu

### Giới thiệu
Bộ dữ liệu mô phỏng hoạt động của một doanh nghiệp thời trang thương mại điện tử tại Việt Nam trong giai đoạn từ **04/07/2012** đến **31/12/2022**. Dữ liệu bao gồm 15 file CSV, được chia thành 4 lớp: **Master** (dữ liệu tham chiếu), **Transaction** (giao dịch), **Analytical** (phân tích) và **Operational** (vận hành).

> **Phân chia dữ liệu cho bài toán dự báo:**
> - `sales_train.csv`: 04/07/2012 → 31/12/2022
> - `sales_test.csv`: 01/01/2023 → 01/07/2024

### Tổng quan các bảng dữ liệu

| # | File | Lớp | Mô tả |
| :--- | :--- | :--- | :--- |
| 1 | `products.csv` | Master | Danh mục sản phẩm |
| 2 | `customers.csv` | Master | Thông tin khách hàng |
| 3 | `promotions.csv` | Master | Các chiến dịch khuyến mãi |
| 4 | `geography.csv` | Master | Danh sách mã bưu chính các vùng |
| 5 | `orders.csv` | Transaction | Thông tin đơn hàng |
| 6 | `order_items.csv` | Transaction | Chi tiết từng dòng sản phẩm trong đơn |
| 7 | `payments.csv` | Transaction | Thông tin thanh toán tương ứng 1:1 với đơn hàng |
| 8 | `shipments.csv` | Transaction | Thông tin vận chuyển |
| 9 | `returns.csv` | Transaction | Các sản phẩm bị trả lại |
| 10 | `reviews.csv` | Transaction | Đánh giá sản phẩm sau giao hàng |
| 11 | `sales.csv` | Analytical | Dữ liệu doanh thu huấn luyện |
| 12 | `sample_submission.csv` | Analytical | Định dạng file nộp bài (mẫu) |
| 13 | `inventory.csv` | Operational | Ảnh chụp tồn kho cuối tháng |
| 14 | `inventory_enhanced.csv` | Operational | Tồn kho mở rộng với các chỉ số dẫn xuất |
| 15 | `web_traffic.csv` | Operational | Lưu lượng truy cập website hàng ngày |

---

### Bảng Master

#### `products.csv` — Danh mục sản phẩm
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `product_id` | int | Khoá chính |
| `product_name` | str | Tên sản phẩm |
| `category` | str | Danh mục sản phẩm |
| `segment` | str | Phân khúc thị trường của sản phẩm |
| `size` | str | Kích cỡ sản phẩm |
| `color` | str | Nhãn màu sản phẩm |
| `price` | float | Giá bán lẻ |
| `cogs` | float | Giá vốn hàng bán |

> **Ràng buộc:** `cogs < price` với mọi sản phẩm.

#### `customers.csv` — Khách hàng
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `customer_id` | int | Khoá chính |
| `zip` | int | Mã bưu chính (FK → `geography.zip`) |
| `city` | str | Tên thành phố của khách hàng |
| `signup_date` | date | Ngày đăng ký tài khoản |
| `gender` | str | Giới tính khách hàng (nullable) |
| `age_group` | str | Nhóm tuổi khách hàng (nullable) |
| `acquisition_channel` | str | Kênh tiếp thị khách hàng đăng ký qua (nullable) |

#### `promotions.csv` — Chương trình khuyến mãi
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `promo_id` | str | Khoá chính |
| `promo_name` | str | Tên chiến dịch kèm năm |
| `promo_type` | str | Loại giảm giá: theo phần trăm hoặc số tiền cố định |
| `discount_value` | float | Giá trị giảm (phần trăm hoặc số tiền tùy `promo_type`) |
| `start_date` | date | Ngày bắt đầu chiến dịch |
| `end_date` | date | Ngày kết thúc chiến dịch |
| `applicable_category` | str | Danh mục áp dụng, `null` nếu áp dụng tất cả |
| `promo_channel` | str | Kênh phân phối áp dụng khuyến mãi (nullable) |
| `stackable_flag` | int | Cờ cho phép áp dụng đồng thời nhiều khuyến mãi |
| `min_order_value` | float | Giá trị đơn hàng tối thiểu để áp dụng khuyến mãi (nullable) |

> **Công thức giảm giá:**
> - `percentage`: `discount_amount = quantity × unit_price × (discount_value/100)`
> - `fixed`: `discount_amount = quantity × discount_value`

#### `geography.csv` — Địa lý
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `zip` | int | Khoá chính (mã bưu chính) |
| `city` | str | Tên thành phố |
| `region` | str | Vùng địa lý |
| `district` | str | Tên quận/huyện |

---

### Bảng Transaction

#### `orders.csv` — Đơn hàng
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `order_id` | int | Khoá chính |
| `order_date` | date | Ngày đặt hàng |
| `customer_id` | int | FK → `customers.customer_id` |
| `zip` | int | Mã bưu chính giao hàng (FK → `geography.zip`) |
| `order_status` | str | Trạng thái xử lý của đơn hàng |
| `payment_method` | str | Phương thức thanh toán được sử dụng |
| `device_type` | str | Thiết bị khách hàng dùng khi đặt hàng |
| `order_source` | str | Kênh marketing dẫn đến đơn hàng |

#### `order_items.csv` — Chi tiết đơn hàng
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `order_id` | int | FK → `orders.order_id` |
| `product_id` | int | FK → `products.product_id` |
| `quantity` | int | Số lượng sản phẩm đặt mua |
| `unit_price` | float | Đơn giá sau khi áp dụng khuyến mãi |
| `discount_amount` | float | Tổng số tiền giảm giá cho dòng sản phẩm này |
| `promo_id` | str | FK → `promotions.promo_id` (nullable) |
| `promo_id_2` | str | FK → `promotions.promo_id`, khuyến mãi thứ hai (nullable) |

#### `payments.csv` — Thanh toán
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `order_id` | int | FK → `orders.order_id` (quan hệ 1:1) |
| `payment_method` | str | Phương thức thanh toán |
| `payment_value` | float | Tổng giá trị thanh toán của đơn hàng |
| `installments` | int | Số kỳ trả góp |

#### `shipments.csv` — Vận chuyển
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `order_id` | int | FK → `orders.order_id` |
| `ship_date` | date | Ngày gửi hàng |
| `delivery_date` | date | Ngày giao hàng đến tay khách |
| `shipping_fee` | float | Phí vận chuyển (0 nếu đơn được miễn phí) |

> **Lưu ý:** Chỉ tồn tại cho đơn hàng có trạng thái `shipped`, `delivered` hoặc `returned`.

#### `returns.csv` — Trả hàng
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `return_id` | str | Khoá chính |
| `order_id` | int | FK → `orders.order_id` |
| `product_id` | int | FK → `products.product_id` |
| `return_date` | date | Ngày khách gửi trả hàng |
| `return_reason` | str | Lý do trả hàng |
| `return_quantity` | int | Số lượng sản phẩm trả lại |
| `refund_amount` | float | Số tiền hoàn lại cho khách |

#### `reviews.csv` — Đánh giá
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `review_id` | str | Khoá chính |
| `order_id` | int | FK → `orders.order_id` |
| `product_id` | int | FK → `products.product_id` |
| `customer_id` | int | FK → `customers.customer_id` |
| `review_date` | date | Ngày khách gửi đánh giá |
| `rating` | int | Điểm đánh giá từ 1 đến 5 |
| `review_title` | str | Tiêu đề đánh giá của khách hàng |

---

### Bảng Analytical

#### `sales.csv` — Dữ liệu doanh thu
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `Date` | date | Ngày đặt hàng |
| `Revenue` | float | Tổng doanh thu thuần |
| `COGS` | float | Tổng giá vốn hàng bán |

---

### Bảng Operational

#### `inventory.csv` — Tồn kho
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `snapshot_date` | date | Ngày chụp ảnh tồn kho (cuối tháng) |
| `product_id` | int | FK → `products.product_id` |
| `stock_on_hand` | int | Số lượng tồn kho cuối tháng |
| `units_received` | int | Số lượng nhập kho trong tháng |
| `units_sold` | int | Số lượng bán ra trong tháng |
| `stockout_days` | int | Số ngày hết hàng trong tháng |
| `days_of_supply` | float | Số ngày tồn kho có thể đáp ứng nhu cầu bán |
| `fill_rate` | float | Tỷ lệ đơn hàng được đáp ứng đủ từ tồn kho |
| `stockout_flag` | int | Cờ báo tháng có xảy ra hết hàng |
| `overstock_flag` | int | Cờ báo tồn kho vượt mức cần thiết |
| `reorder_flag` | int | Cờ báo cần tái đặt hàng sớm |
| `sell_through_rate`| float | Tỷ lệ hàng đã bán so với tổng hàng sẵn có |
| `product_name` | str | Tên sản phẩm |
| `category` | str | Danh mục sản phẩm |
| `segment` | str | Phân khúc sản phẩm |
| `year` | int | Năm trích từ `snapshot_date` |
| `month` | int | Tháng trích từ `snapshot_date` |

#### `web_traffic.csv` — Lưu lượng truy cập
| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `date` | date | Ngày ghi nhận lưu lượng |
| `sessions` | int | Tổng số phiên truy cập trong ngày |
| `unique_visitors` | int | Số lượt khách truy cập duy nhất |
| `page_views` | int | Tổng số lượt xem trang |
| `bounce_rate` | float | Tỷ lệ phiên chỉ xem một trang rồi thoát |
| `avg_session_duration_sec` | float | Thời gian trung bình mỗi phiên (giây) |
| `conversion_rate` | float | Tỷ lệ phiên dẫn đến đặt hàng |
| `traffic_source` | str | Kênh nguồn dẫn traffic về website |

---

### Quan hệ giữa các bảng

| Quan hệ | Cardinality |
| :--- | :--- |
| `orders` ↔ `payments` | 1 : 1 |
| `orders` ↔ `shipments` | 1 : 0 hoặc 1 (trạng thái shipped/delivered/returned) |
| `orders` ↔ `returns` | 1 : 0 hoặc nhiều (trạng thái returned) |
| `orders` ↔ `reviews` | 1 : 0 hoặc nhiều (trạng thái delivered, ~20%) |
| `order_items` ↔ `promotions` | nhiều : 0 hoặc 1 |
| `products` ↔ `inventory` | 1 : nhiều (1 dòng/sản phẩm/tháng) |

---

## 2. Đề Bài

### Phần 1 — Câu hỏi Trắc nghiệm
Chọn một đáp án đúng nhất cho mỗi câu hỏi. Các câu hỏi yêu cầu tính toán trực tiếp từ dữ liệu được cung cấp.

**Q1. Trong số các khách hàng có nhiều hơn một đơn hàng, trung vị số ngày giữa hai lần mua liên tiếp (inter-order gap) xấp xỉ là bao nhiêu? (Tính từ `orders.csv`)**
A) 30 ngày
B) 90 ngày
C) 180 ngày
D) 365 ngày

**Q2. Phân khúc sản phẩm (`segment`) nào trong `products.csv` có tỷ suất lợi nhuận gộp trung bình cao nhất, với công thức `(price − cogs)/price`?**
A) Premium
B) Performance
C) Activewear
D) Standard

**Q3. Trong các bản ghi trả hàng liên kết với sản phẩm thuộc danh mục Streetwear (join `returns` với `products` theo `product_id`), lý do trả hàng nào xuất hiện nhiều nhất?**
A) defective
B) wrong_size
C) changed_mind
D) not_as_described

**Q4. Trong `web_traffic.csv`, nguồn truy cập (`traffic_source`) nào có tỷ lệ thoát trung bình (`bounce_rate`) thấp nhất trên tất cả các ngày xuất hiện nguồn đó trong cột `traffic_source`?**
A) organic_search
B) paid_search
C) email_campaign
D) social_media

**Q5. Tỷ lệ phần trăm các dòng trong `order_items.csv` có áp dụng khuyến mãi (tức là `promo_id` không null) xấp xỉ là bao nhiêu?**
A) 12%
B) 25%
C) 39%
D) 54%

**Q6. Trong `customers.csv`, xét các khách hàng có `age_group` khác null, nhóm tuổi nào có số đơn hàng trung bình trên mỗi khách hàng cao nhất? (tổng số đơn / số khách hàng trong nhóm)**
A) 55+
B) 25–34
C) 35–44
D) 45–54

**Q7. Vùng (`region`) nào trong `geography.csv` tạo ra tổng doanh thu cao nhất trong `sales_train.csv`?**
A) West
B) Central
C) East
D) Cả ba vùng có doanh thu xấp xỉ bằng nhau

**Q8. Trong các đơn hàng có `order_status = 'cancelled'` trong `orders.csv`, phương thức thanh toán nào được sử dụng nhiều nhất?**
A) credit_card
B) cod
C) paypal
D) bank_transfer

**Q9. Trong bốn kích thước sản phẩm (S, M, L, XL), kích thước nào có tỷ lệ trả hàng cao nhất, được định nghĩa là số bản ghi trong `returns` chia cho số dòng trong `order_items` (join với `products` theo `product_id`)?**
A) S
B) M
C) L
D) XL

**Q10. Trong `payments.csv`, kế hoạch trả góp nào có giá trị thanh toán trung bình trên mỗi đơn hàng cao nhất?**
A) 1 kỳ (trả một lần)
B) 3 kỳ
C) 6 kỳ
D) 12 kỳ

---

### Phần 2 — Trực quan hoá và Phân tích Dữ liệu
Khám phá bộ dữ liệu để tìm ra các insight có ý nghĩa kinh doanh. Phần này được đánh giá dựa trên **tính sáng tạo, chiều sâu phân tích** và **chất lượng trình bày**.

#### Yêu cầu
1. **Trực quan hoá (Visualizations):** Tạo các biểu đồ, đồ thị, bản đồ hoặc dashboard trực quan để thể hiện các pattern, xu hướng và mối quan hệ trong dữ liệu. Mỗi hình ảnh cần có tiêu đề, nhãn trục rõ ràng và chú thích phù hợp.
2. **Phân tích (Analysis):** Viết phần giải thích đi kèm mỗi trực quan hoá, bao gồm:
    - Mô tả những gì biểu đồ thể hiện và tại sao góc nhìn này quan trọng.
    - Các phát hiện chính (key findings) được hỗ trợ bởi số liệu cụ thể.
    - Ý nghĩa kinh doanh (business implications) hoặc đề xuất hành động (actionable recommendations).

#### Tiêu chí đánh giá Phần 2:
- **Descriptive:** What happened? (Thống kê tổng hợp chính xác, biểu đồ đúng).
- **Diagnostic:** Why did it happen? (Giả thuyết nhân quả, so sánh phân khúc, xác định bất thường).
- **Predictive:** What is likely to happen? (Ngoại suy xu hướng, phân tích tính mùa vụ).
- **Prescriptive:** What should we do? (Đề xuất hành động kinh doanh được hỗ trợ bởi dữ liệu).

---

### Phần 3 — Mô hình Dự báo Doanh thu (Sales Forecasting)

#### Bối cảnh kinh doanh
Cần dự báo nhu cầu chính xác ở mức chi tiết để tối ưu hoá phân bổ tồn kho, lập kế hoạch khuyến mãi và quản lý logistics trên toàn quốc.

#### Định nghĩa bài toán
Dự báo cột `Revenue` trong khoảng thời gian của `sales_test.csv`. Mỗi dòng trong tập test là một bộ `(Date, Revenue, COGS)` duy nhất trong giai đoạn **01/01/2023 – 01/07/2024**.

#### Chỉ số đánh giá
1. **Mean Absolute Error (MAE):** $MAE = \frac{1}{n} \sum_{i=1}^{n} |F_i - A_i|$
2. **Root Mean Squared Error (RMSE):** $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (F_i - A_i)^2}$
3. **R² (Coefficient of Determination):** $R^2 = 1 - \frac{\sum_{i=1}^{n} (A_i - F_i)^2}{\sum_{i=1}^{n} (A_i - \bar{A})^2}$

> $F_i$: giá trị dự báo, $A_i$: giá trị thực, $\bar{A}$: trung bình giá trị thực.
> **MAE và RMSE càng thấp càng tốt. R² càng cao càng tốt.**

#### Ràng buộc
1. **Không dùng dữ liệu ngoài:** Tất cả đặc trưng phải được tạo từ các file dữ liệu được cung cấp.
2. **Tính tái lập (Reproducibility):** Đính kèm toàn bộ mã nguồn. Đặt random seed khi cần thiết.
3. **Khả năng giải thích (Explainability):** Trong report, bao gồm mục giải thích các yếu tố dẫn động doanh thu (VD: feature importances, SHAP values...).

---

## 3. Thang điểm Chấm thi

| Phần | Nội dung | Điểm | Tỷ trọng |
| :--- | :--- | :--- | :--- |
| 1 | Câu hỏi Trắc nghiệm (MCQ) | 20 | 20% |
| 2 | Trực quan hoá & Phân tích (EDA) | 60 | 60% |
| 3 | Mô hình Dự báo Doanh thu | 20 | 20% |
| **Tổng** | | **100** | **100%** |

### Chi tiết thang điểm Phần 2 (60 điểm)
- **Chất lượng trực quan hoá (15đ):** Biểu đồ chuẩn, thẩm mỹ, rõ ràng.
- **Chiều sâu phân tích (25đ):** Bao phủ 4 cấp độ phân tích (Descriptive -> Prescriptive).
- **Insight kinh doanh (15đ):** Phát hiện có giá trị thực tiễn, đề xuất khả thi.
- **Tính sáng tạo & kể chuyện (5đ):** Góc nhìn độc đáo, mạch trình bày coherent.

### Chi tiết thang điểm Phần 3 (20 điểm)
- **Hiệu suất mô hình (12đ):** Xếp hạng Leaderboard (MAE, RMSE, R²).
- **Báo cáo kỹ thuật (8đ):** Chất lượng pipeline (FE, CV, leakage), giải thích mô hình.

---

## 4. Hướng dẫn Nộp bài

Ngoài file `submission.csv` trên Kaggle, mỗi đội cần nộp đầy đủ các thành phần sau:

### 1. Kết quả dự báo (Kaggle)
- **Nộp tại:** [https://www.kaggle.com/competitions/datathon-2026-round-1](https://www.kaggle.com/competitions/datathon-2026-round-1)
- **Yêu cầu:** File `submission.csv` phải đảm bảo đúng số dòng và đúng thứ tự như file `sample_submission.csv`.

### 2. Báo cáo (Report PDF)
- **Định dạng:** Sử dụng template LaTeX của **NeurIPS**.
- **Độ dài:** Tối đa **4 trang** (không tính phần References và Appendix).
- **Nội dung bắt buộc:**
    - Trực quan hoá và phân tích dữ liệu (Phần 2).
    - Pipeline mô hình và kết quả thực nghiệm (Phần 3).
    - Link GitHub repository của nhóm.

### 3. GitHub Repository
- **Chế độ:** Public (hoặc cấp quyền truy cập cho BTC trước deadline).
- **Nội dung:** Chứa toàn bộ mã nguồn (source code), Notebook xử lý và file submission.
- **Tài liệu:** Phải có file `README.md` mô tả chi tiết cấu trúc thư mục và hướng dẫn cụ thể cách chạy lại code để tái lập kết quả.

### 4. Form nộp bài chính thức
Các đội điền đầy đủ các thông tin sau vào Form của BTC:
- Đáp án 10 câu hỏi trắc nghiệm (Phần 1).
- Tệp báo cáo (PDF).
- Link GitHub repository.
- Link Kaggle submission.
- Ảnh thẻ sinh viên của **tất cả** các thành viên trong nhóm.

> [!IMPORTANT]
> **Thời hạn quan trọng:** Nhóm thi cam kết ít nhất 1 thành viên có thể tham gia trực tiếp Vòng Chung kết vào ngày **23/05/2026** tại Đại học VinUni, Hà Nội.
