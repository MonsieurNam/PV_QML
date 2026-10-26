# Photonic PV pilot — quyết định dataset và cấu hình đề xuất

Ngày rà soát: 2026-10-06. Phạm vi: targeted source review và đối chiếu code hiện tại, không phải systematic literature review. Đây là đề xuất nghiên cứu, chưa phải protocol đã được duyệt hoặc kết quả trên dữ liệu thật.

## 1. Quyết định

Chọn **NIST Campus PV — Roof Tilted Array, dữ liệu một phút năm 2015–2016**, cho pilot đầu tiên. Chọn **DKASC Alice Springs, Array 24 Q CELLS**, làm ứng viên kiểm chứng ngoài site hoặc dự phòng khi không thể tải/đạt QC NIST. Quyết định dựa trên metadata và độ phù hợp với hợp đồng dữ liệu, chưa dựa trên WIS của bất kỳ mô hình nào.

Mục tiêu vẫn là dự báo xác suất công suất AC trung bình 5 phút tại chân trời 15 phút. Kiến trúc là **classical lag/window memory + photonic feature map + classical quantile readout**. Chưa có QPU: làm simulator feasibility, không gọi đây là physical quantum reservoir có bộ nhớ, hoặc bằng chứng quantum advantage.

Ưu tiên nghiên cứu của Quanova được giữ nguyên: use case công nghiệp, baseline mạnh, latency và chi phí có thể kiểm chứng; build chaotic microcomb là nhánh phần cứng riêng.

## 2. Câu hỏi chọn dữ liệu và sàng lọc

Điều kiện bắt buộc: AC power đo thật ở 1 hoặc 5 phút; GHI và ambient temperature đo thật; metadata công suất, thời gian và sensor; đủ độ dài để tách thời gian; nguồn gốc/điều khoản sử dụng có thể truy vết. Không dùng upsampling 15 phút thành 5 phút; không dựng power từ irradiance rồi coi là generation đo thật; không thay onsite observations bằng retrospective weather mà không đổi protocol.

| Dataset | Bằng chứng từ nguồn sở hữu/tác giả | Quyết định cho pilot hiện tại |
|---|---|---|
| NIST Campus PV | Public portal; one-minute data 2015–2016; AC meter, weather, data dictionary và issue logs | **Ưu tiên 1**, chọn Roof; vẫn phải kiểm tra raw files |
| DKASC Alice Springs | AC power là trung bình 5 phút của đo 1 giây; có metadata từng array và notes vận hành | **Ưu tiên 2**, Array 24 là ứng viên; cần acquisition/QC |
| BR-PVGen, công bố 2026 | 51 plant, SCADA và onsite weather; dữ liệu đã chuẩn hóa 15 phút, có smoothing/interpolation | Giữ cho nghiên cứu khác ở cadence 15 phút; không tương thích target hiện tại |
| KIT MPVBench, cập nhật 2026 | Power/temperature 5 phút; README không liệt kê GHI; profile 1c ngừng báo, sau đó được ghi 0 | Không chọn khi GHI là mandatory; có thể dùng cho protocol power-only riêng |
| ARENA Solar Power Ensemble Forecaster | Catalog mô tả đo 1 phút từ SCADA/weather tại solar farms, bộ khoảng 6 tháng | Watchlist industrial; chưa kiểm tra raw schema/license/access, không thay thế mặc định |

Nguồn: [NIST testbeds](https://www.nist.gov/el/beed/heat-transfer-alternative-energy-systems/photovoltaic-testbeds), [DKASC glossary](https://dkasolarcentre.com.au/glossary), [BR-PVGen](https://zenodo.org/records/21511487), [MPVBench](https://github.com/KIT-IAI/MPVBench), [ARENA catalog](https://researchdata.edu.au/arena-solar-power-final-dataset/1608240?source=suggested_datasets).

Không ưu tiên dataset chỉ vì mới xuất bản. Dữ liệu NIST cũ phù hợp để kiểm thử phương pháp; không đại diện sẵn cho nhà máy PV Việt Nam hay toàn ngành.

## 3. NIST: cấu hình dữ liệu cụ thể

### 3.1 Site và kênh đo

Site ID đề xuất trong canonical data: `nist_roof`.

- Tọa độ: latitude 39.1354, longitude -77.2156.
- Inverter rated AC power: **75 kW**. Không lấy panel DC rating khoảng 73 kW làm AC clipping bound.
- Native cadence: **1 phút**, aggregate thành 5 phút bằng mean của đủ 5 quan sát duy nhất, không nội suy missing bins.
- Giờ nguồn: **Eastern Standard Time cố định UTC−05:00, không DST**. Dùng `Etc/GMT+5` nếu timestamp naive đúng như tài liệu; nếu file đã có offset thì parse offset thực tế, không gắn timezone lần nữa. Không dùng `America/New_York` tự đổi DST.

Tài liệu NIST công bố hai gián đoạn lớn của Ground năm 2015. Vì vậy Roof được chọn trước Ground để giảm khối lượng xử lý downtime; điều này không chứng minh Roof không có lỗi. Phải đọc issue log đi kèm khi tải. [NIST data-descriptor và Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7339723/)

| Vai trò canonical | Kênh nguồn dự kiến từ dictionary | Ghi chú cần nghiệm thu trên CSV |
|---|---|---|
| `timestamp` | `TIMESTAMP` / timestamp field trong export | Xác nhận tên thực tế, interval start/end và fixed EST |
| `power_kw` | `PwrMtrP_kW_Avg` | AC real power từ meter; không dùng kVA, DC power hoặc accumulated kWh |
| `ghi_w_m2` | `RefCell2_Wm2_Avg` ở Roof | GHI từ silicon reference cell nằm ngang, cleaned; ghi rõ loại sensor |
| `temperature_c` | `AmbTemp_C_Avg` | Ambient ngoài trời, không phải module/inverter temperature |
| `site_id` | Thêm hằng `nist_roof` trong adapter | Site identity không phụ thuộc tên file tùy ý |
| `capacity_kw` | Thêm metadata AC 75.0 | Nameplate là rated power; phải kiểm tra upper-bound policy |

Tên ở bảng là **mapping đề xuất dựa trên dictionary và quy tắc suffix `_Avg`**, chưa xác minh trên raw CSV. GHI silicon reference cell không đồng nhất với thermopile pyranometer băng rộng; đó là giới hạn đo lường phải công bố. Không map `Pyra*_mV` sang W/m² nếu chưa có calibration. [NIST data dictionary](https://www.nist.gov/document/datadictionarysupplementalcontentpdf)

Các fault/status channels và issue log được giữ trong sidecar QC, không đưa thành feature biết trước sự cố tương lai. Một fault flag không tự chứng minh power measurement sai: sự cố inverter có thể là generation thật cần giữ. Meter/communication errors và outage phải được phân loại riêng.

### 3.2 Split chốt trước khi xem model performance

| Split | Khoảng theo EST cố định | Cutoff UTC cho config hiện tại |
|---|---|---|
| Train | 2015-01-01 đến 2015-12-31 | `2016-01-01T04:59:59Z` |
| Validation | 2016-01-01 đến 2016-06-30 | `2016-07-01T04:59:59Z` |
| Test | 2016-07-01 đến 2016-12-31 | `2017-01-01T04:59:59Z` |

Không random split. Tuning, scaling và resource decisions chỉ dùng train/validation; purge target vượt ranh giới split. Test có 184 ngày lịch trước QC; chưa biết số sample/ngày hợp lệ sau QC. H2 holdout chưa đủ để tuyên bố hiệu quả cả năm; cần site/period độc lập khi đi tiếp.

### 3.3 Acquisition và quyền dùng

Portal chính thức: <https://pvdata.nist.gov/>; chọn location Roof và bulk download dữ liệu numeric của 2015, 2016, kèm dictionary/issue log. Không cần lấy sky images, spectral curves hay 1-second archives cho G0–G3. Portal hiện sử dụng giao diện dynamic; đã xác nhận trang chính thức và metadata, **chưa tải numeric archives hoặc xác minh trực tiếp headers/checksums**.

Dataset citation: Boyd M, Chen T, Dougherty B (2017), *NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets*, DOI [10.18434/M3S67G](https://doi.org/10.18434/M3S67G).

Giữ acknowledgement, notices và ghi rõ mọi phép biến đổi; không tự gán CC BY cho NIST. Đối chiếu license đi kèm archive với [NIST licensing policy](https://www.nist.gov/director/licensing) trước khi phân phối lại dữ liệu. Đề xuất lưu original archives read-only và ghi URL, access date, SHA256, size, schema, acquisition errors trong manifest.

## 4. Latency: hai thí nghiệm khác nhau

### A. Offline ideal-availability — có thể triển khai trên contract hiện tại sau adapter/QC

Tại issue time `s`, dùng bin kết thúc tại `s`, coi available latency là 0 trong **mô hình benchmark offline**. Target là mean AC của bin `(s+10, s+15]`, không phải mean toàn bộ 15 phút tiếp theo.

Ví dụ: issue 12:00, latest observation bin kết thúc 12:00, target bin kết thúc 12:15. Metadata archive chưa cung cấp availability SLA; latency 0 không phải thông tin được đo từ vận hành.

Đây là feasibility gate về feature/readout. Kết quả tốt ở A chưa đủ quyết định triển khai công nghiệp.

### B. Operational proxy — đề xuất kế tiếp, cần sửa contract

Chọn giả định engineering **5 phút availability delay**, chưa coi là latency thật của NIST hay nhà máy. Issue theo lưới 5 phút tại `s`; anchor `u=s−5` là bin mới nhất đã available. Target vẫn kết thúc tại `s+15`, tức `u+20`.

- Input 4-mode: `[P(u), P(u−15), GHI(u), Temp(u)]`.
- Input 6-mode nếu được duyệt trước test: `[P(u), P(u−5), P(u−15), P(u−60), GHI(u), Temp(u)]`.
- Strong baseline power lags đều tính từ `u`; target tính từ **issue time `s`**.
- Chỉ dùng observation có `available_at <= s`; không dịch nhãn nguồn rồi làm mất nghĩa timestamp.
- 0/5/10 phút là các scenario có thể đăng ký trước, không lựa chọn scenario tốt nhất sau holdout.

`splits.py` hiện gán `issued_at=interval_end` và bắt buộc lag-0/current weather có sẵn. Đổi YAML thành latency 5 phút mà không đổi feature construction sẽ loại các sample lag-0. **B chưa chạy đúng chỉ bằng configuration**. Protocol vận hành cần phiên bản mới được phê duyệt; không chèn các key không được schema hỗ trợ vào YAML hiện tại.

## 5. Cấu hình mô hình và tài nguyên

| Thành phần | Đề xuất |
|---|---|
| Simulator | Perceval 1.3.0 trong lockfile; `pcvl.BackendFactory.get_backend("SLOS")`; verify class/module thực tế |
| Primary size | **2 photons, 4 modes**; output Fock distribution dimension `C(5,2)=10` |
| Expansion | 6 modes / 21 outputs chỉ khi profile đạt và scope được freeze trước test; 3 photons tắt mặc định |
| Encoding | Train-only MinMax; clip inputs vào [0,1]; `phi=c*x_scaled`, `c` trong `{pi/2, pi}` chọn bằng validation |
| Readout | Quantile linear regression; alpha `{1e−4,1e−3,1e−2,1e−1}` |
| Quantiles | `{0.05,0.10,0.25,0.50,0.75,0.90,0.95}` |
| Baselines | Persistence; matched raw-feature quantile regression; matched RFF (10 components với 4 modes); strong lag-XGBoost quantile |
| RFF | Gamma `{0.1,1,10}`; matched input/scaler ownership và random-state được ghi lại |
| XGBoost | Depth `{3,6}`, learning rate `{0.03,0.1}`, 300 trees, `hist`, `n_jobs=1`; grid giữ như plan |
| Physical control | Distinguishable photons, cùng circuit/input/resource accounting; không chỉ so persistence |
| Seeds | Giữ master `20260928`; 5 circuit seeds, namespace riêng RFF/XGB/bootstrap/sampling, lưu manifest |
| CPU budget | 1 worker; chốt BLAS/OpenMP thread settings; hard ceiling 24 GiB RSS và 12 giờ/stage theo config hiện tại |
| GPU/QPU | **Chưa thuê GPU; chưa cần QPU** để làm pilot exact nhỏ này |
| Finite-shot | 1,000/10,000 raw attempts là diagnostic/proposal stage sau, không trộn vào kết luận exact G3 |

Theo [Perceval backend documentation](https://perceval.quandela.net/docs/v1.3/backends.html), SLOS hỗ trợ full probability distribution phù hợp feature extractor. Không khẳng định SLOS luôn nhanh nhất cho mọi input/size; đo workload cụ thể.

Phase mapping giới hạn miền và tránh cuộn vòng; không bảo đảm feature hữu ích hoặc ánh xạ toàn cục injective. Cần log tỷ lệ input vượt train range rồi bị clip. Không fit lại scaler bằng test để giảm clip rate.

Code hiện tạo V1/V2 bằng NumPy RNG có seed riêng; không cần gọi thêm Perceval random-circuit API vốn không được dùng. Các stochastic APIs nào thực sự chạy phải nhận seed được quản lý. Chronological split là deterministic, không cần random seed để chia split.

5 circuit seeds là 5 lặp kiến trúc; không chọn seed tốt nhất sau test. Trung bình WIS qua seeds không phải prediction ensemble. Nếu triển khai ensemble 5 circuits thì phải đăng ký cách gộp prediction và tính đủ 5 circuit calls.

Profile trước full run: (i) feature extraction trên 1,000 input train với cả controls/scales, (ii) quantile fitting tại 2k/5k/10k train samples, (iii) cache/serialization/RSS. Toàn bộ hai năm có khoảng 210k bin 5 phút trước QC/daylight; không suy từ fixture nhỏ rằng full run sẽ xong trong 12 giờ. Bottleneck có thể là quantile solver, không chỉ permanent. Nếu cần training subsampling, đăng ký quy tắc trên train và sửa sample-contract trước; không âm thầm lấy ít test hoặc đổi dataset theo score.

## 6. QC, evaluation và gate

### G0/G1 — đề xuất nghiệm thu dữ liệu

1. Lấy raw file, dictionary, issue log, license; xác minh schema, units, clock, interval-end semantics và AC metadata. Chưa đạt thì **HOLD/BLOCKED về dữ liệu**, chưa đánh giá khoa học.
2. Kiểm tra duplicate timestamps, NAN/sentinels, meter/communication errors, physical range và các block liên tục. Giữ genuine cloud ramps/clipping; không lọc theo forecast error. Không dùng cutoff GHI tương lai để làm sample dễ hơn.
3. Daylight theo solar elevation tại tọa độ Roof, không dùng power>0 để âm thầm loại trip. Target truth không được clip chỉ để phù hợp capacity.
4. Capacity 75 kW là rated AC metadata. Nếu valid readings hoặc inverter specifications chứng minh upper bound cần khác, phải tách normalization capacity và prediction clipping bound trước khi test; không xóa observations hợp lệ để giữ bound 75.
5. Timebox cleaning 10 engineer-hours. Khi chạm giới hạn, freeze QC policy, giữ các block đủ điều kiện, ghi exclusions; không bù các khoảng khuyết lớn bằng interpolation. Không bảo đảm đủ samples chỉ vì metadata nói availability cao.
6. Đề xuất coverage gate trước test: giữ >=90% các bin daylight kỳ vọng mỗi split; >=150 usable calendar test days; >=20 usable **non-overlapping** 7-day calendar blocks với definition coverage đã khóa. Đây là ngưỡng thiết kế đề xuất, không phải kết quả đã tính hoặc tính độc lập thống kê được chứng minh. Nếu không đạt, EXTEND QC hoặc đổi acquisition theo metadata, đăng ký lại trước model test.

### G2/G3 — đề xuất nghiệm thu nghiên cứu

- Chọn baseline B bằng validation WIS trên timestamp/weighting chung; freeze B trước test. Code hiện mặc định comparison model là XGBoost, cần sửa nếu B được hiểu là best classical comparator.
- Đồng nhất validation objective: hiện classical timestamp-average và quantum day-average có thể cho lựa chọn khác nhau khi số samples/day khác nhau. Đề xuất timestamp-average chung; day-average là sensitivity.
- WIS primary; coverage/width của khoảng 90% và MAE phụ. `S=1−WIS_system/WIS_B`, denominator phải dương.
- Bootstrap paired calendar blocks 1 ngày, sensitivity 2/7 ngày, 2,000 replicates. Giữ common timestamp masks và dependencies; không coi overlapping blocks là các tuần độc lập. 5 seeds không làm tăng số ngày thông tin thực.
- Tách quantum-only trên successful timestamps và system có fallback trên toàn eligible timestamps. Tính B trên đúng mỗi tập tương ứng; báo success/fallback rates và toàn bộ failed-attempt costs.
- Ngưỡng đề xuất theo plan: delta=2%, coverage 90% trong [85%,95%], resource gate đạt. Schema hiện khóa các giá trị này; cần phê duyệt thực chất, không tự điền `approved_by/approved_at`.
- **GO tới stage sau:** cận dưới CI 95% của system skill >2% cùng calibration/resource gates đạt. Muốn gán đóng góp cho quantum interference phải vượt physical controls với phân tích đã đăng ký.
- **EXTEND:** CI chưa phân định được ngưỡng và có thí nghiệm cụ thể trong budget bổ sung; thêm dữ liệu test mới/độc lập nếu đã dùng holdout, không retune rồi reuse holdout như confirmatory.
- **STOP scope này:** cận trên CI <2% và không có lợi ích chi phí/nghiệp vụ khác đã đăng ký. Không kết luận toàn bộ QML không hiệu quả.
- Data invalid, protocol chưa duyệt, pipeline lỗi hoặc chưa chạy đúng latency là **technical/protocol blockage**, không phải scientific STOP.

## 7. DKASC: kế hoạch kiểm chứng thứ hai

Ứng viên: **Array 24 Q CELLS, fixed ground mount, 2016**; nguồn ghi inverter SMA SMC 6000A và panel rating 6.1 kW. Không tự dùng 6.1 kWp DC làm AC bound. Cần datasheet/inverter continuous AC rating và current raw metadata trước khi điền capacity. [Array metadata](https://dkasolarcentre.com.au/source/alice-springs/dka-m16-a-phase-1)

Khung thời gian đề xuất để kiểm tra metadata: train 2022; validation H1 2023; test H2 2023. **Chưa freeze như protocol**: cần xác minh schema, timestamp/cadence, thời tiết và ảnh hưởng các notes trong đúng period. Alice Springs dùng `Australia/Darwin` nếu timestamps được xác nhận là giờ địa phương UTC+09:30.

Notes xác nhận weather fault khoảng 2020-03-06 và sửa power supply ngày 2020-03-10; không coi lỗi này kéo dài toàn bộ những năm sau. Metering issue 2021 có thể tạo missing generation dù hệ vẫn phát điện; không biến thành zero target. [Notes 2020](https://dkasolarcentre.com.au/download/notes-on-the-data/p4), [notes 2021](https://dkasolarcentre.com.au/download/notes-on-the-data/p3).

Download portal yêu cầu tên/email/organization và chấp nhận điều khoản; chưa gửi form hoặc đại diện Quanova đồng ý. Chủ dữ liệu yêu cầu tải một file mỗi lần, 5-minute mỗi năm khoảng 7 MB và có thể mất khoảng 10 phút. [DKASC download](https://www.dkasolarcentre.com.au/download?location=alice-springs)

Đây là kiểm chứng ngoài site/khí hậu, không phải dataset để tìm score dương sau một test âm. Site Alice Springs khô nóng và Roof NIST ôn đới không thay thế validation trên SCADA thực của đối tác Việt Nam.

## 8. Việc cần làm tiếp theo, theo thứ tự

| Việc | Input | Output | Chặn tiến độ nào? |
|---|---|---|---|
| Acquire NIST Roof numeric files/logs | Portal public, 2015/2016 | Raw manifest, headers, checksum, notices | Data chưa về thì G1 chưa nghiệm thu |
| Nghiệm thu dữ liệu + adapter | Raw/dictionary/logs | Canonical CSV/Parquet, QC/exclusion report, UTC verification | Không đúng target/units/clock thì HOLD |
| Khóa offline protocol | QC và proposed split/config | Real config + human approval, chưa tự phê duyệt | Chưa duyệt thì không confirmatory GO/STOP |
| Chỉnh evaluation contract | Best-B, common weighting, mask policy | Tests + review cho comparators/evaluation | Không nên chạy final comparison với nghĩa B sai |
| Profile G2 | Train-only representative inputs | CPU/RSS/solver estimate, scope freeze [4] hay [4,6] | Không tăng photons để thử tùy hứng |
| Run G3 exact | Locked data/config/code | Baseline/control/seed/WIS/calibration/CI/resource report | Chỉ dữ liệu thật + protocol hợp lệ mới quyết định khoa học |
| Operational-proxy contract | Issue grid, delay scenarios | Availability-aware features + target tests | Sau đó mới đánh giá feasibility latency |

Không có GPU purchase, code implementation, protocol approval, numeric archive download hay model training được thực hiện trong lượt rà soát này. Trạng thái phù hợp: **dataset/config recommendation ready; raw-data acceptance pending; operational scenario requires implementation**.

## 9. Evidence trail và giới hạn rà soát

Đã đối chiếu publisher documentation, data-descriptor primary source, dataset repository/README, Perceval v1.3 documentation và các file local `config.py`, `data.py`, `splits.py`, `photonics.py`, `seeds.py`, `baselines.py`, `evaluation.py` cùng stage-gate plan. Dictionary PDF được kiểm tra bằng MarkItDown; không dùng excerpt chat làm bằng chứng về raw data.

Chưa xác nhận: file-level completeness, CSV header thực tế, timestamp bin convention trong export, AC meter calibration áp dụng từng period, issue-log đầy đủ, native availability timestamps, legal classification trong archive, NIST current bulk-download success và DKASC acquisition. Đó là các task acceptance, không phải các giả định được mặc định đúng.

Không đưa tỷ lệ missing, model scores, runtime dữ liệu thật hoặc novelty/publication claims vào kết luận khi chưa đo. Bảng lựa chọn thay cho hình minh họa; workflow AI schematic không có OpenRouter credential trong môi trường và không cần thiết để chốt dataset/config.
