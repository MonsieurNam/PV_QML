# Kế hoạch và nội dung báo cáo 10 slide

Deck 16:9; nội dung tiếng Anh; speaker notes tiếng Việt; dự kiến trình bày 8–10 phút. Phong cách chalkboard với palette navy–teal–amber theo xác nhận. Yêu cầu bổ sung hình thực tế được áp dụng vào slide 2 và 7; số slide giữ nguyên.

| Slide | Thông điệp | Hình/bằng chứng |
| --- | --- | --- |
| 1 | Photonic PV forecasting: từ câu hỏi công nghiệp đến pilot | Sơ đồ PV history → photonic features → quantile forecasts |
| 2 | Ba điều kiện để chọn hướng nghiên cứu | Ảnh đúng NIST Roof Tilted Array từ NIST |
| 3 | Literature dẫn đến thay đổi thiết kế | Ba luồng paper → bài học: baseline, cơ chế, tài nguyên |
| 4 | Vì sao chọn PV và giả thuyết có thể bác bỏ | Câu hỏi nghiên cứu; ranh giới exact hiện tại và bước tương lai |
| 5 | Kiến trúc đề xuất: input → các nhánh mô hình → output | Mạch photonic, control photon phân biệt, raw/RFF, XGBoost, persistence; đầu ra 7 quantile; metric và chú thích input mở rộng của XGBoost |
| 6 | Giá trị ứng dụng và giao thoa là hai phép kiểm tra | Đối chứng; WIS và công thức skill; coverage/width, runtime/RAM; confidence interval và tiêu chí gate |
| 7 | Dữ liệu, chronological split và target | Đồ thị 25 quan trắc train thật, issue 12:00 và target 12:10–12:15 |
| 8 | Dữ liệu đã qua audit khả dụng | Biểu đồ coverage thật; 157 ngày và 35 tuần test đủ chuẩn |
| 9 | Phần đã xong và nút thắt hiện tại | Checklist tiến độ; tài nguyên ngoại suy có nhãn rõ |
| 10 | Bước thực nghiệm kế tiếp và điều kiện đi tiếp | Verify capacity → run locked models → evaluate holdout |

Quy trình thực thi: chuẩn bị nguồn → xác nhận brief/thiết kế trên PPT Master → khóa Design Spec → hand-author SVG → kiểm tra trang đầu và toàn deck → đối chiếu chart với số liệu → bổ sung ảnh/plot thực theo yêu cầu → rà soát hình thức → viết và nhúng notes → xuất SVG preview và PPTX → kiểm tra package.

Nguồn dữ liệu ví dụ được chọn cố định trước khi xem trace: NIST train, 01/06/2015, 11:00–13:00 fixed EST. Đường nét đứt là quan trắc về sau để minh họa target, không phải dự báo. Không mở hiệu năng holdout và không huấn luyện mô hình trong quá trình làm báo cáo. CSV, scripts và hash được lưu trong dự án thuyết trình.

Ảnh site lấy từ trang NIST Roof Tilted Array chính thức: https://www.nist.gov/el/beed/heat-transfer-alternative-energy-systems/photovoltaic-testbeds/roof-tilted-array . Hồ sơ nguồn ảnh và kết quả đối chiếu chính sách dùng ảnh nằm tại images/image_sources.json. Không gán giấy phép CC cho NIST.

Tài liệu chi tiết: design_spec.md; spec_lock.md; notes/total.md; sources/project_brief_vi.md; validation/chart_verification.json; sources/train_example_provenance.json.
