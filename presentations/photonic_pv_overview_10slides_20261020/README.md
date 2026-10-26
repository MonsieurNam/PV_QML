# Photonic PV: báo cáo nghiên cứu 10 slide

Báo cáo 16:9, 1280 × 720; nội dung slide tiếng Anh, lời thuyết trình tiếng Việt.
PPTX và sơ đồ kiến trúc được cập nhật ngày 23/10/2026. Các nhận định về tiến độ
thực thi khoa học dùng mốc bằng chứng 16/10/2026.

## Tài liệu sử dụng

- [PowerPoint chỉnh sửa được](exports/Photonic_PV_Project_Overview_10slides.pptx).
- [Lời thuyết trình chi tiết cho từng slide](notes/presenter_script_detailed_vi.md).
- [Giải thích Fock, V1/V2, D, quantile head và metric](notes/architecture_and_metrics_explained_vi.md).
- [Speaker notes nhúng trong PowerPoint](notes/total.md).
- [Hình kiến trúc](exports/Proposed_Architecture.png) và [mô hình so sánh, metric](exports/Comparators_and_Metrics.png).
- [Kế hoạch nội dung](content_plan.md), [thiết kế](design_spec.md) và [execution lock](spec_lock.md).

## Nguồn và khả năng tái lập

`svg_output/` chứa 10 trang SVG gốc dùng để xuất thành các đối tượng DrawingML
chỉnh sửa được. `images/` giữ ảnh NIST đã chọn, plot quan trắc train và hồ sơ nguồn.
`sources/` chứa danh mục tài liệu, số liệu audit, 25 quan trắc train minh họa,
ước lượng tài nguyên và provenance/hash. `scripts/` chứa mã trích xuất và vẽ ví dụ.
`validation/` giữ báo cáo kiểm tra SVG, chart và file xuất.

Ví dụ quan trắc lấy từ train ngày 01/06/2015; nét đứt trên plot là quan trắc tương
lai để giải thích target. Target là trung bình bin 5 phút kết thúc ở t+15, không
phải trung bình cả 15 phút tiếp theo. Không có điểm số mô hình thật trong deck.

File Parquet NIST đầy đủ được lưu trong `artifacts/` tại môi trường thực nghiệm;
đường dẫn và hash có trong `sources/train_example_provenance.json`. CSV minh họa
đã được lưu để có thể xem và vẽ lại ví dụ mà không cần mở dữ liệu holdout.

Workflow dựng lại là PPT Master: kiểm tra `svg_output/` theo `spec_lock.md`, tách
`notes/total.md`, tạo SVG preview tự chứa rồi xuất native PPTX. PPT Master là công
cụ cài riêng, không được đóng gói trong repository này. Preview, phiên UI, log và
bản xuất trùng được loại khỏi Git bằng `.gitignore` cục bộ.
