# Đánh giá và hoàn thiện pilot photonic QML cho dự báo điện mặt trời

Ngày rà soát: 26/09/2026. Bối cảnh: Quanova chưa có quyền truy cập QPU; đây là nhánh nghiên cứu phụ, không phải dự án điều khiển lưới điện.

## 1. Kết luận điều hành

Bản đề xuất mới đáng triển khai ở quy mô pilot, nhưng chưa sẵn sàng chuyển nguyên văn thành sprint. Cần sửa mô hình dự báo xác suất, thiết kế encoding, cách đếm ngân sách đo và kiểm định tính mới. Không có kết quả thực nghiệm nào được tạo trong lần đánh giá này.

Tên làm việc đề xuất:

**Đánh giá đặc trưng lượng tử quang học có xét tổn hao cho dự báo xác suất công suất PV trước 15 phút dưới ngân sách đo hữu hạn và dịch chuyển phân phối.**

Tên tiếng Anh: *Loss-aware photonic quantum features for probabilistic 15-minute-ahead PV forecasting under finite measurement budgets and distribution shift.*

Đây là một câu hỏi thực nghiệm, không phải lời hứa về quantum advantage. Tính mới khả dĩ nằm ở tương tác giữa giao thoa nhiều photon, cách xử lý loss, ngân sách đo và chất lượng dự báo xác suất khi chuyển miền; không nằm ở việc ghép quantum features với Ridge hoặc chuyển sang dữ liệu Việt Nam.

## 2. Phạm vi và giới hạn tìm kiếm

Đây là tổng quan có mục tiêu, không phải systematic review/PRISMA và không chứng minh rằng chưa có ai thực hiện tổ hợp trên. Ưu tiên công trình 2025–26, bổ sung tài liệu nền tảng cần thiết. Loại các công trình sau 26/09/2026, quantum-inspired nhưng không có mô hình lượng tử liên quan, và nghiên cứu tối ưu hóa PV không trực tiếp trả lời câu hỏi dự báo.

Nguồn được đối chiếu: trang nhà xuất bản Nature Portfolio, Quantum, Springer và Elsevier; arXiv; kỷ yếu PMLR/Optica; tài liệu chính thức Quandela/Perceval, XGBoost, EPRI và NREL. Crossref được tìm kiếm và dùng xác minh DOI. Truy vấn Semantic Scholar bị giới hạn HTTP 429, không được tính là tìm kiếm thành công.

Nhóm truy vấn tiêu biểu: `photonic quantum machine learning solar forecasting finite shots distribution shift 2026`; `photonic quantum reservoir computing distinguishability loss finite sampling`; `quantum probabilistic photovoltaic forecasting`; `Mitigating photon loss in linear optical quantum circuits`; `photonic quantum forecasting September 2026`; `adaptive conformal time series distribution shift`.

Một số bài Elsevier chỉ đọc được abstract/metadata; không suy diễn chất lượng kiểm định hoặc kết quả chi tiết chưa tiếp cận. Các preprint được ghi riêng. Danh mục nguồn và trạng thái đọc nằm trong [tài liệu tham khảo](sources/Photonic_PV_pilot_20260926_references.md).

## 3. Bằng chứng mới làm thay đổi kế hoạch như thế nào?

| Nhánh bằng chứng | Phát hiện cần lưu ý | Hệ quả cho Quanova |
|---|---|---|
| Finite-shot QRC | Ahmed và cộng sự đã phân tích nhiễu lấy mẫu và cách huấn luyện QRC hữu hạn mẫu năm 2025 [1] | Chỉ quét 1.000/10.000 shots không đủ mới |
| Giao thoa và chuỗi thời gian | Thực nghiệm 2026 dùng hai photon, so sánh distinguishable/indistinguishable, với feedback để tạo memory [2] | Phải có ablation vật lý; không gán memory cho giao thoa đơn thuần |
| Loss mitigation | Mills–Mezher, Quantum 2026: dùng recycled probabilities từ lượt bị mất photon; estimator có bias nhưng có thể giảm tổng sai số [3] | So sánh hậu tuyển với tận dụng lượt có loss; không tự nhận phát minh recycling |
| Hướng Quandela | Preprint tháng 5/2026 có thí nghiệm Ascella, huấn luyện có perturbation để giảm sim-to-hardware mismatch [4] | Noise-aware training đã có tiền lệ; hardware shift khác weather shift |
| Nhiều mode hơn | Nghiên cứu GBS 2026 chỉ ra lợi ích so với các biến thể cổ điển hóa của cùng hệ, không tuyên bố thắng mọi thuật toán cổ điển [5] | Tách lợi ích vật lý, lợi ích ML và lợi ích kinh tế |
| Quantum PV | Bài Solar Energy 2025 đã nghiên cứu QML cho dự báo công suất PV [6] | “Dùng QML dự báo solar” không phải đóng góp mới |
| Classical PV | Renewable Energy và Information Sciences tháng 8/2026 đã nghiên cứu xác suất, physics-informed và non-crossing quantiles [7,8] | Không so QML xác suất với chỉ các baseline dự báo điểm |
| Hiệu chỉnh khi chuyển miền | UAI 2026 có phương pháp calibration model-agnostic theo predict-then-update [9] | Cần tách hiệu quả của calibration khỏi hiệu quả quantum features |

Hai cảnh báo bổ sung: preprint về photonic forecasting tài chính 2026 tự ghi rõ latency không gồm feature extraction và test chỉ sáu ngày [10]. Preprint boson-sampling ML 2025 báo cáo lợi ích trong cả chế độ photon phân biệt được [11]. Vì vậy, nonlinear features có ích chưa đồng nghĩa phần lợi ích ấy cần tính lượng tử.

Không có bằng chứng từ các nguồn đã kiểm tra cho phép khẳng định photonic features chắc chắn cải thiện CRPS trên PV Việt Nam hoặc cải thiện OOD ở ngân sách cụ thể.

## 4. Các điểm cần sửa trong đề xuất

### 4.1. Gọi đúng kiến trúc và vị trí bộ nhớ

Sliding window cổ điển cộng với feature map quang học không có state tái hồi nên gọi là **photonic quantum feature model / QELM với temporal embedding**, hoặc ghi rõ memory hoàn toàn cổ điển nếu vẫn dùng thuật ngữ QRC theo nghĩa rộng.

LOI tĩnh không chỉ là “hash function”: nó là một phép biến đổi vật lý tạo feature map phụ thuộc encoding và measurement. Nó có thể hữu ích mà không có bộ nhớ nội tại. Cũng cần bỏ cụm “hồ chứa tính toán cực nhanh”: tốc độ chưa được đo.

### 4.2. Ridge không tự tạo dự báo xác suất

Ridge thông thường tạo một giá trị điểm. Randomness của quantum shots không phải phân phối có điều kiện của công suất tương lai. Vẽ khoảng từ nhiều lần chạy QPU không tự tạo khoảng dự báo thời tiết đã được hiệu chuẩn.

Pilot nên giữ Ridge làm diagnostic dự báo điểm; đầu ra chính là **linear multi-quantile readout có regularization**, dùng cùng kiến trúc cho các feature maps được so sánh. Tập quantile đề xuất 0,05 đến 0,95; xử lý quantile crossing bằng một quy tắc được chốt trước. XGBoost có objective quantile riêng [12].

Dùng pinball loss và WIS làm metric trực tiếp từ quantiles. Chỉ gọi là CRPS khi có predictive CDF/ensemble hợp lệ, hoặc ghi rõ CRPS xấp xỉ qua tích phân quantile, bao gồm quy ước nội suy và hai đuôi. Một khoảng 90% không đủ xác định toàn bộ phân phối.

Nếu bổ sung conformal calibration, áp dụng cùng quy trình cho các mô hình đối chứng. Conformal thông thường không có bảo đảm tự động dưới phụ thuộc thời gian và arbitrary shift; cần tách coverage dài hạn với coverage theo mùa/ramp [13]. Calibration chỉ cập nhật sau khi nhãn đã tới.

### 4.3. Chặn leakage bằng thời điểm sẵn có, không bằng việc cấm tương lai

Cho phép bản tin dự báo thời tiết cho t+15 nếu đã phát hành và thực sự có thể nhận trước thời điểm ra dự báo t. Cấm quan trắc thực tế ở t+15 và bản dự báo được chỉnh sửa/phát hành sau t. Nếu không có lịch sử issue/availability timestamps, bỏ NWP ở pilot đầu.

EPRI có cả baseline NWP và persistence [14]; sử dụng thông tin tương lai đã dự báo hợp lệ không phải leakage. Lập bảng `event_time`, `issued_at`, `available_at` cho mỗi nguồn.

### 4.4. Encoding phải tạo tín hiệu thật, không chỉ tồn tại trong simulator

Khởi đầu với Fock input cố định và phase encoding giữa hai lớp trộn:

`Fock input → V₁ cố định → D(phi(x)) → V₂ cố định → detection → features`.

Với |n₁,…,nₘ⟩, nếu đặt D ngay trước mọi phép trộn thì D|n⟩ = exp(iΣ nⱼphiⱼ)|n⟩: chỉ là pha toàn cục. Xác suất đếm photon không đổi theo x. Đây là suy luận trực tiếp từ tác dụng của phase shifter, và phải trở thành unit test trước benchmark. Đặt phase ngay trước phép đo số photon cũng không tự thay đổi xác suất đếm.

Không đưa arbitrary amplitude encoding vào sprint đầu: khả năng khởi tạo StateVector trong simulator không chứng minh trạng thái đa photon ấy chuẩn bị được bằng phần cứng/passive optics hiện có. Bắt đầu với phase encoding, Fock states và mạch có thể biên dịch; nghiên cứu amplitude encoding chỉ khi đã xác định chi phí và thủ tục state preparation.

Permanent không phải “tích thường”; đó là hàm ma trận xuất hiện trong biên độ chuyển trạng thái boson. Tính khó mô phỏng ở các chế độ nhất định không tự suy ra generalization tốt hoặc advantage ở hệ vài photon.

### 4.5. Photon loss chưa phải regularization có ích

Phải định nghĩa loss model và quy tắc giữ/loại mẫu trước khi đưa ra giả thuyết. Trong mô hình đơn giản: n photon đầu vào xác định, pure loss đồng đều độc lập với mode, truyền qua mỗi photon τ, number-resolving detection lý tưởng, chỉ giữ lượt còn đủ n photon, không có lỗi khác:

`p(s | còn đủ n photon, x) = p_ideal(s | x)` và `p_accept = τⁿ`.

Trong chế độ này, tăng loss không làm trơn feature distribution đã điều kiện hóa; nó làm ít lượt hợp lệ hơn. Nếu giữ số accepted samples không đổi, chi phí tăng nhưng expected features không đổi. Với loss không đồng đều, multiphoton contamination, dark counts hoặc filter khác, kết luận này không còn được dùng nguyên xi.

Tách ba phép thử:

1. Cố định raw attempts: trả lời chi phí triển khai thật trong mô hình nguồn đã định nghĩa.
2. Cố định accepted samples: diagnostic chất lượng feature, không phải so sánh chi phí công bằng.
3. Giữ lượt có loss hoặc mitigation: feature distribution/estimator thay đổi; đánh giá cả bias, variance và chi phí xử lý.

Perceval phân biệt `max_shots`, `max_samples`, `physical_perf`, `logical_perf`; không trộn chúng trong bảng kết quả [15]. “Shot” ở simulator còn cần ánh xạ sang source trial/clock cycle của máy thật, không tự đồng nhất.

### 4.6. Cần đối chứng cùng cấu trúc, không chỉ ESN/XGBoost

Hai nhóm so sánh có mục tiêu khác nhau:

| Nhóm | Mô hình | Câu hỏi |
|---|---|---|
| Giá trị ứng dụng | Persistence; clear-sky persistence và persistence ensemble; quantile XGBoost; ESN với quantile head | Có ích hơn mô hình cổ điển phù hợp không? |
| Quy nguồn lợi ích | Raw lags + cùng head; random Fourier/polynomial features + cùng head; photon phân biệt được; photon không phân biệt được | Lợi ích đến từ encoding, feature expansion hay giao thoa nhiều photon? |

Ở nhóm quy nguồn: giữ cùng cửa sổ, preprocessing, readout, ngân sách tuning và số feature khi khả thi; báo cả matched-dimension và matched-cost nếu hai tiêu chí không thể đồng thời khớp. Nếu nối raw features vào quantum features, classical controls cũng phải được nối tương tự.

Smart persistence nên được định nghĩa bằng clear-sky normalization cụ thể. Ngoại suy gradient là một baseline khác, không dùng tên thay thế mơ hồ. Điều chỉnh solar geometry không cần quan trắc thời tiết tương lai.

### 4.7. Bổ sung noise đúng với photonics

Finite-shot sampling là độ bất định do đo hữu hạn, không cùng loại với loss hoặc phase drift. Pipeline tối thiểu cần khả năng kiểm tra transmittance, distinguishability và sai lệch phase/unitary; brightness, g² và detector response tùy máy mục tiêu [16].

Không quét toàn bộ tích Descartes mọi tham số ngay từ đầu. Chốt một circuit size nhỏ, một encoding, hai chế độ indistinguishability; sau đó mới mở rộng noise. Chưa có calibration thật thì gọi là sensitivity study với noise giả định, không gọi digital twin đã xác nhận của Quandela.

### 4.8. OOD phải tách khỏi thời gian và random seeds

Train mùa nắng/test mùa mưa là một stress test, chưa phải chứng minh OOD tổng quát. Cần nhiều chronological origins, kiểm tra season shift và site transfer riêng; ghi rõ sample support/đặc điểm thời tiết đã thay đổi. Hardware drift là một trục khác.

Fit scaler/PCA, feature selection, readout và calibration chỉ từ dữ liệu đúng phạm vi training/validation. Đặt nhãn training kết thúc trước cutoff; không nội suy hai chiều qua ranh giới split. Lịch sử trước cutoff trong cửa sổ test là hợp lệ nếu thực sự sẵn có.

Chạy nhiều circuit seeds và shot seeds để đo bất định số học. Đánh giá độ bất định hiệu năng bằng paired block bootstrap theo ngày/khối đủ dài hoặc phương pháp phù hợp tự tương quan; không coi hàng nghìn timestamp hoặc nhiều seed là các nhà máy độc lập. Chốt một primary contrast; các phân tích phụ cần kiểm soát multiple comparisons.

### 4.9. Dữ liệu phải có độ phân giải và bản chất phù hợp

Chọn công suất AC đo mỗi 1–5 phút, có metadata timestamp, capacity, sensor latency, timezone và quality flags. Phải chốt target: ví dụ công suất trung bình của khoảng 5 phút kết thúc tại t+15, khác với điện năng trong toàn bộ 15 phút tiếp theo.

Không lấy dataset hourly rồi nội suy thành 5 phút để tuyên bố học được biến thiên ngắn hạn. NREL Solar Power Data for Integration Studies có dữ liệu 5 phút nhưng là các nhà máy mô phỏng, không phải đo thực địa [17]. Dùng được cho phát triển pipeline, phải ghi rõ giới hạn và chưa xác nhận giá trị ở Việt Nam.

Tách đánh giá daylight/all-hours để đêm không che lấp sai số. Phân biệt công suất thực phát và công suất khả dụng; curtailment, clipping, outages, sensor failures phải có quy tắc xử lý trước, không loại sau khi xem mô hình nào thua. Định nghĩa ramp threshold trên train hoặc theo tiêu chí nghiệp vụ, không chọn lại trên test.

### 4.10. Không dùng ESN thất bại làm bằng chứng dừng quantum

ESN thua persistence có thể đơn giản là đặc tính bài toán, không chứng minh dữ liệu sai hay quantum chắc chắn vô ích. Nó nhắc phải xác định headroom và ngưỡng cải thiện có giá trị. Kiểm toán baseline trước, sau đó dành ngân sách nhỏ kiểm tra feature map lượng tử thay vì mở một sweep lớn.

Ngược lại, mô phỏng lý tưởng thắng chưa chứng minh phần cứng thắng. Mô phỏng finite-shot không thể đo latency E2E của cloud thật. Chỉ báo thời gian simulator và phân tích kịch bản hardware có giả định minh bạch; không lấy latency readout làm latency dự báo. NVQLink của Quandela là một kiến trúc tích hợp riêng, không bảo đảm quyền truy cập hay latency cho Quanova [18].

## 5. Thiết kế nghiên cứu sửa đổi

### Câu hỏi chính

Ở cùng dữ liệu và raw measurement budget, đặc trưng từ photon không phân biệt được có cải thiện proper scoring rule trên weather-shift test so với photon phân biệt được và các classical feature controls hay không?

### Câu hỏi phụ ưu tiên

Với loss cố định, hậu tuyển đủ photon, feature giữ lượt có loss và recycling mitigation khác nhau thế nào về calibration–accuracy–cost? Đây là bài toán downstream forecasting, không chỉ phục hồi phân phối lý tưởng.

Giả thuyết “loss giúp regularization” chỉ giữ làm exploratory. Không lấy nó làm cơ sở biện minh cho hardware. Không mở đồng thời nghiên cứu topology, weak measurement, optical feedback và amplitude encoding.

### Kiến trúc và control points

| Bước | Đầu ra | Kiểm soát bắt buộc |
|---|---|---|
| Dữ liệu đã sẵn có tại t | Lag window + biến lịch/solar geometry | Availability audit |
| Compression tùy chọn | Vector nhỏ | Chỉ fit train; control dùng cùng compression |
| V₁–D(x)–V₂ | Distribution/counts | Test dữ liệu có thực sự thay đổi output |
| Detection + post-processing | Feature vector | Cố định raw budget; log accepted counts |
| Regularized quantile head | Quantiles công suất | Cùng head cho feature controls |
| Calibration tùy chọn | Quantiles/intervals đã hiệu chỉnh | Predict trước, update sau khi nhãn tới |
| Evaluation | WIS/CRPS, coverage, cost | Frozen tests + paired uncertainty |

### Thí nghiệm tối thiểu khả thi

- Bắt đầu 2 photon, 4–6 mode; đây là kích thước debug đề xuất, không phải tuyên bố tốt nhất. Tăng lên 3 photon/6–8 mode khi profiling cho phép.
- Dùng một window chính được chọn bằng validation; không mặc định cần 24 giờ ở mẫu 1 phút. Các biến chu kỳ có thể tách khỏi window.
- Exact probabilities là diagnostic reference, không tự xem là cận trên của generalization.
- Raw budgets ban đầu: 10², 10³, 10⁴; 10⁵ chỉ nếu cần tìm giới hạn. Giữ cùng ngân sách cho các circuit ensembles bằng cách cộng tổng shots.
- Train/test: exact/exact; exact/finite; finite/finite. Case exact-training là giả định triển khai phải được ghi rõ, gồm chi phí mô phỏng/offline training.
- Lossless và một số mức loss giả định; ideal vs degraded indistinguishability; một phép stress phase mismatch. Mở rộng tuần tự, không brute force mọi kết hợp.
- Primary endpoint: daylight mean WIS trên các blocked out-of-time tests, cùng quantile grid. CRPS khi CDF đã định nghĩa. Secondary: MAE, calibration tại 80/90%, sharpness, ramp Brier score và hiệu năng từng site/mùa.
- Báo raw attempts, accepted counts, tỷ lệ chấp nhận, số mạch, thời gian feature extraction, preprocessing, readout, tổng training/tuning và memory. Tách overhead mô phỏng khỏi giả định hardware.

Mục tiêu công nghiệp cần một người sử dụng xác định: forecast phục vụ quyết định nào, update cadence, chi phí sai số hai phía và data SLA. Chưa có các thông tin ấy thì output chỉ là feasibility benchmark, chưa là xác nhận nhu cầu khách hàng.

## 6. Khung 8–12 tuần và điều kiện nguồn lực

Ước tính dưới đây dành cho hai kỹ sư gần toàn thời gian, chuyên gia năng lượng hỗ trợ đều, và dataset đã tiếp cận được. Không bao gồm thời gian xin dữ liệu/quota, chờ hardware, hoặc thời gian phản biện tạp chí.

| Thời gian | Việc chính | Điều kiện qua mốc |
|---|---|---|
| Tuần 1–2 | Audit dữ liệu, target, availability, persistence/quantile XGBoost; đọc và tái hiện mạch nhỏ | Dataset card; split manifest; không leakage; encoding test |
| Tuần 3–4 | ESN/random features; point và quantile readouts; lý tưởng distinguishable/indistinguishable | Biết baseline mạnh nhất; có matched-feature controls |
| Tuần 5–6 | Raw-shot budgets, loss và postselection; sanity checks bảo toàn xác suất | Resource logs và synthetic tests đúng |
| Tuần 7–8 | Weather/site shift; repeat seeds; frozen evaluation; một cơ chế loss-aware | Có kết luận sơ bộ với uncertainty |
| Tuần 9–12 | Chỉ khi cần: recycling, phase mismatch, refinement; báo cáo và reproduction package | Đánh giá go/no-go có điều kiện, không chọn test đẹp nhất |

Nếu thực tế chỉ một người làm nhánh phụ khoảng 10–15 giờ/tuần, nên đặt mục tiêu **4–6 tuần cho baseline audit + một mạch lý tưởng nhỏ**. Full protocol sẽ dài hơn đáng kể; không áp mốc 8–12 tuần của đội ba người lên lịch cá nhân.

## 7. Go/no-go không ép kết quả thành thắng/thua

Chốt ngưỡng cải thiện thực dụng với anh Quang/người dùng trước khi mở test. Ví dụ “2% cải thiện proper score” chỉ là ngưỡng thử để thảo luận, không phải chuẩn ngành hay kết quả dự báo trước.

- **Go sang hardware feasibility:** lợi ích lặp lại trên nhiều khối/mùa, vượt ngưỡng đã chốt với uncertainty phù hợp; shot/loss budget khả thi; phần lợi ích không biến mất khi so controls đúng. QPU thật vẫn cần kiểm chứng.
- **Giữ làm nghiên cứu vật lý:** chỉ thắng distinguishable counterpart nhưng không thắng baseline ML mạnh. Không thương mại hóa thành quantum advantage.
- **Chuyển hướng cổ điển:** classical random features giữ nguyên lợi ích. Đây có thể là pipeline hữu ích nhưng không chứng minh nhu cầu QPU.
- **No-go cho use case hiện tại:** mô hình thua rõ, budget vượt giới hạn hoặc dữ liệu/encoding không phù hợp.
- **Inconclusive:** confidence interval quá rộng, quá ít ngày/site hoặc các mô hình tương đương. Không gọi là chứng minh thất bại hay thành công.

Negative result tốt có ích nội bộ; khả năng thành bài báo phụ thuộc tính tổng quát, độ chặt chẽ và insight, không được bảo đảm. Cũng không tự động chuyển sang weak measurement/topology chỉ vì forecasting thất bại: đó là câu hỏi và tài nguyên khác.

## 8. Các điều cần hỏi Quandela trước giai đoạn hardware

1. Máy, số mode/photon, loại detector và các giới hạn circuit thực sự có thể dùng?
2. Có lấy raw events/counts trước photon-number filtering không; sample/shot trong API và quota tính thế nào?
3. Có dữ liệu calibration về brightness, transmittance, indistinguishability, g², phase drift và detector errors không?
4. Có chạy batch nhiều input-dependent circuits không; chi phí compile, reconfiguration và queueing?
5. Có thể lưu loss sectors/mẫu bị filter để nghiên cứu recycling hay chỉ nhận mẫu đã hậu tuyển?
6. Điều kiện công bố kết quả, tái lập code và chia sẻ calibration?

Không cần chờ các câu trả lời này để bắt đầu audit dữ liệu và simulation nhỏ; nhưng không cam kết hardware-compatible trước khi xác minh.

## 9. Bản mô tả ngắn để gửi anh Quang

> Em đề xuất pilot photonic QML cho dự báo xác suất công suất điện mặt trời trước 15 phút. Giai đoạn chưa có QPU sẽ mô phỏng một feature extractor quang học nhỏ, dùng bộ nhớ cửa sổ cổ điển và quantile readout. Em sẽ so với persistence, XGBoost, ESN và các feature maps cổ điển; kiểm tra riêng đóng góp của giao thoa nhiều photon. Điểm nghiên cứu ưu tiên là chất lượng/calibration khi thay đổi thời tiết dưới cùng ngân sách đo, tính cả photon loss và hậu tuyển. Nếu kết quả có ý nghĩa, mình mới xin quota Quandela để xác minh sai khác mô phỏng–phần cứng; chưa đặt mục tiêu quantum advantage hay tốc độ vượt trội.

Ghi chú phương pháp: kỹ năng đánh giá khoa học được dùng để tách giả thuyết, bằng chứng, nhiễu và tiêu chí dừng; kỹ năng tổng quan dùng để cập nhật prior art và kiểm tra DOI. Dịch vụ tạo scientific schematic thiếu cấu hình nên báo cáo dùng bảng kiến trúc, không có hình AI và không có kết quả thực nghiệm minh họa giả.
