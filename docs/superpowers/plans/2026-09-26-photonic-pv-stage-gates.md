# Photonic PV Pilot — Implementation Plan ở mức khung

> **For agentic workers:** REQUIRED SUB-SKILL khi được yêu cầu triển khai: dùng `executing-plans` để thực hiện theo từng mốc; chỉ dùng `subagent-driven-development` khi được phép phân công song song. Checklist dưới đây là kế hoạch, không phải bằng chứng công việc đã chạy.

**Goal:** Xác định có nên đầu tư tiếp vào đặc trưng lượng tử quang học cho dự báo xác suất công suất PV trước 15 phút, dựa trên hiệu quả dự báo, đóng góp vật lý và ngân sách đo.

**Architecture:** Bộ nhớ cửa sổ cổ điển → feature map cổ điển hoặc Fock–V1–phase encoding–V2 → quantile readout cổ điển. Đầu tiên mô phỏng nhỏ, sau đó mới kiểm tra loss và QPU theo quyết định riêng. Không điều khiển lưới điện.

**Tech Stack:** Python, Perceval, NumPy/pandas, scikit-learn, XGBoost, pytest; phiên bản sẽ được khóa khi triển khai, không giả định môi trường đã cài đủ.

---

Ngày lập: 26/09/2026. Trạng thái: đề xuất để thống nhất phạm vi và tiêu chí trước thí nghiệm. Người quyết định đầu tư: anh Quang. Đây là khung quản trị nghiên cứu theo yêu cầu người dùng, chưa phải kế hoạch code chi tiết từng hàm. Không tạo worktree, cài package, tải dataset hay chạy mô phỏng trong lần lập kế hoạch này.

Cập nhật phản biện kỹ thuật: bổ sung timebox dữ liệu, kiểm soát phase encoding, trần mô phỏng và tách quantum-only/system evaluation. Các thay đổi này không tăng ngân sách 96 giờ của G0–G3; chưa có benchmark thực nghiệm.

## 1. Mục tiêu, phạm vi và giả định

Hai kết quả phải được phân biệt:

1. **Giá trị ứng dụng:** có cải thiện so với baseline cổ điển mạnh trong điều kiện đo hữu hạn không?
2. **Đóng góp vật lý:** có bằng chứng phần cải thiện liên quan tới giao thoa nhiều photon không?

Thắng đối chứng vật lý nhưng thua baseline ML không phải thành công thương mại. Thắng ML nhưng không hơn photon phân biệt được không chứng minh cần giao thoa nhiều photon. Một kết quả pilot âm không bác bỏ toàn bộ QML.

Phạm vi pilot: một site/dataset chính, horizon 15 phút, một encoding, 2 photon và 4–6 mode khởi đầu. Không thêm topology search, weak measurement, amplitude encoding, optical feedback, HFT hoặc điều khiển lưới điện.

Giả định nguồn lực để lập lịch: 2 kỹ sư × 12 giờ/tuần × 4 tuần = **96 giờ kỹ sư**; người hỗ trợ năng lượng 1–2 giờ/tuần. Nếu nguồn lực khác, giữ gates và điều chỉnh lịch. Pilot có thể chỉ đạt feasibility, chưa đủ kết luận OOD tổng quát.

## 2. Input và output của hệ thống dự báo

### Input

- `site_id`, timestamp có timezone, công suất AC, capacity và quality flags.
- Dữ liệu thật độ phân giải 1–5 phút; metadata cho biết giá trị tức thời hay trung bình khoảng, timestamp đầu hay cuối khoảng.
- Các biến tùy có: irradiance, nhiệt độ, lịch/solar geometry. Không bắt buộc thêm gió nếu không có ý nghĩa với site.
- `available_at` hoặc giả định độ trễ được ghi rõ cho từng nguồn. Không dùng quan trắc tương lai. NWP chỉ dùng nếu biết bản tin đã sẵn có tại thời điểm dự báo.
- Window lịch sử được chọn bằng validation, preprocessing fit trên train.
- Cấu hình thí nghiệm: split, model/circuit seed, readout, raw-shot budget, noise và detector/postselection policy.

### Nhãn

`y_t = công suất AC trung bình trong khoảng 5 phút kết thúc tại t+15 phút`.

Đây là target mặc định để tránh nhập nhằng. Nếu nghiệp vụ cần điện năng của toàn bộ 15 phút tới, phải đổi target và version protocol trước khi chạy. Nhãn mỗi mẫu là một giá trị đo, không phải các quantile tính từ cửa sổ tương lai.

### Output mỗi lần dự báo

- `issued_at`, `target_interval_start/end`, `site_id`, `model_version`.
- Các quantile q05, q10, q25, q50, q75, q90, q95; khoảng dự báo 50%, 80%, 90%.
- Quality/status flag; không tự điền một dự báo thành công khi input lỗi.
- Thời gian preprocessing, feature extraction, readout, tổng thời gian; raw attempts, accepted counts, số circuit khi có sampling.
- Giá trị về miền công suất hợp lệ và quantile không giao nhau theo quy tắc chốt trước, áp dụng nhất quán cho mọi mô hình.
- Nếu sampling không đủ mẫu hoặc dự báo lỗi: dùng fallback F đã chốt ở G0, gắn cờ và ghi nguyên nhân. Mặc định F = baseline B được chọn bằng validation. Nếu dùng F khác B, báo rõ cả hai.
- Bắt buộc báo riêng quantum-only trên các timestamp quantum thành công và system (quantum + fallback) trên toàn bộ timestamp đủ điều kiện đánh giá. Mỗi score có baseline trên đúng tập timestamp tương ứng; quantum-only là đánh giá có điều kiện, không thay thế metric ứng dụng. Chi tiết ở mục 5.

### Output cuối pilot

1. Dataset card và split manifest có thể tái lập.
2. Bộ kiểm thử dữ liệu, vật lý và metric.
3. Baseline cùng quantum feature controls và predictions đã lưu.
4. Bảng accuracy–calibration–resource, uncertainty và negative results.
5. Decision memo: GO / EXTEND / STOP / BLOCKED, bằng chứng, phạm vi kết luận và ngân sách đề nghị tiếp theo.

## 3. Input của dự án: những gì cần chốt ở G0

| Hạng mục | Mặc định đề xuất | Ai chốt |
|---|---|---|
| Mục tiêu ứng dụng | Feasibility dự báo; chưa tự động vận hành hệ thống | Anh Quang + người hỗ trợ năng lượng |
| Dataset | Một nguồn công suất PV đo thực tế; kiểm toán trước khi chọn chính thức | Kỹ sư A |
| Nhân lực và thời gian | 96 giờ kỹ sư / 4 tuần theo giả định trên | Anh Quang |
| Chi phí ngoài máy sẵn có | Không mua dịch vụ/quota; đo runtime local trước | Anh Quang nếu phát sinh |
| Compute cap | Khởi đầu 2 photon, 4–6 mode; trần pilot 3 photon; khóa wall-clock/RAM tối đa sau profiling trước sweep | Nhóm đề xuất, anh Quang duyệt khi đổi scope |
| Success/fallback policy | Khóa accepted-count tối thiểu, điều kiện output hợp lệ, timeout và fallback trước test; SLA vận hành chốt trước G5 | Nhóm + người hỗ trợ năng lượng |
| Cải thiện thực dụng tối thiểu δ | 2% relative WIS skill là điểm khởi đầu thảo luận, không phải chuẩn ngành | Anh Quang, chốt trước mở test |
| Khả năng hiệu chuẩn | Coverage 90% trong khoảng 85–95% là ngưỡng sàng lọc minh họa | Nhóm + người hỗ trợ năng lượng |
| Thử bổ sung | Tối đa 1 đợt, 24 giờ kỹ sư, phải được duyệt | Anh Quang |

Các con số là tiêu chí quản trị đề xuất, chưa phải yêu cầu khách hàng. Thiếu lựa chọn ngưỡng không được coi là đã thông qua GO; có thể chuẩn bị dữ liệu/tests nhưng phải khóa protocol trước đánh giá xác nhận.

## 4. Các giai đoạn: input → công việc → output → gate

### G0 — Chốt đề bài và tiêu chí (4 giờ kỹ sư; đầu tuần 1)

**Input:** đề xuất nghiên cứu, nguồn lực thật, yêu cầu anh Quang.

**Việc cần làm:**

- [ ] Chốt target và mục đích sử dụng dự báo; ghi rõ nếu chưa có người dùng công nghiệp.
- [ ] Chốt resource cap, primary metric, baseline selection rule, thresholds, chế độ train/test exact hay finite, raw budget chính và phạm vi được phép kết luận.
- [ ] Chỉ định kỹ sư A phụ trách dữ liệu/baseline, B phụ trách photonics; cả hai kiểm tra chéo leakage và evaluation.
- [ ] Ghi cách xử lý test reuse và quyền duyệt các lần mở rộng.

**Output:** `artifacts/protocol_v1.md`, `configs/pilot.yaml`, `artifacts/budget.csv`.

**Gate:** GO khi scope và budget rõ; EXTEND để giải quyết một lựa chọn cụ thể; BLOCKED nếu chưa có người quyết định/nguồn lực. Không làm sweep khi G0 chưa chốt.

### G1 — Dữ liệu và classical baseline (20 giờ; tuần 1–2)

**Input:** protocol, dataset được phép sử dụng và metadata.

**Việc cần làm:**

- [ ] Kiểm tra units, timestamp, missingness, duplicates, clipping, outages/curtailment, license và nguồn gốc đo thật/mô phỏng.
- [ ] Tạo target và feature chỉ dùng dữ liệu sẵn có; quyết định quy tắc loại/đánh dấu trước khi so mô hình.
- [ ] Timebox cứng **10 giờ kỹ sư cho làm sạch dữ liệu, nằm trong 20 giờ G1**, không cộng thêm. Đến hạn: khóa QC policy, giữ các khối ngày/tuần liên tục đạt tiêu chí và ghi exclusion manifest; không xóa dữ liệu gốc. Nếu phần còn lại không đủ tạo splits hợp lệ, chuyển gate dữ liệu thay vì tiếp tục làm sạch vô hạn.
- [ ] Phân biệt lỗi đo/ghi dữ liệu với biến động vật lý thật. Mây che, ramp nhanh, clipping, inverter trip hoặc curtailment không tự động là dữ liệu hỏng khi target là công suất AC thực tế. Chỉ loại theo phạm vi nghiệp vụ và QC đã chốt, không theo volatility cao hoặc sai số mô hình.
- [ ] Không nội suy gap lớn; ngưỡng gap và cách xử lý gap nhỏ phải khóa theo cadence/metadata trước so mô hình. Không dùng thông tin tương lai để điền feature, không dùng nhãn nội suy làm ground truth test. Cửa sổ lịch sử và target phải thuộc khối hợp lệ, không nối qua khoảng khuyết bị loại.
- [ ] Báo tỷ lệ loại/missing và lý do theo thời gian, mùa và các chế độ thời tiết/ramp khi có nhãn tin cậy. Nếu lọc mất chế độ khó, giới hạn kết luận về đúng miền dữ liệu giữ lại; không tuyên bố robustness toàn miền.
- [ ] Chia train/validation và giữ ít nhất một khoảng thời gian cuối chưa đụng tới; fit transformations chỉ từ train.
- [ ] Chạy persistence, clear-sky persistence nếu có metadata đủ, quantile XGBoost; thêm baseline xác suất persistence/residual với hiệu chỉnh từ dữ liệu quá khứ.
- [ ] Đánh giá daylight riêng; chốt baseline mạnh B bằng validation và cùng ngân sách tuning.

**Output:** `artifacts/data_card.md`, `artifacts/qc_exclusions.csv`, `artifacts/splits.json`, `artifacts/baseline_validation.csv`, ledger thời gian làm sạch, kiểm thử leakage, gap boundaries và target alignment.

**Gate:**

- GO: dữ liệu đủ tạo train/validation/test độc lập về thời gian; không còn lỗi leakage đã biết; baseline chạy tái lập.
- EXTEND: lỗi metadata/gap có khả năng xử lý hoặc cần một nguồn dữ liệu thay thế trong giới hạn giờ.
- BLOCKED: chưa có quyền dữ liệu hoặc chưa nhận được file, không được diễn giải là thất bại khoa học.
- STOP pilot hiện tại: sau hạn xử lý vẫn không có dataset dùng được; hoặc target không phù hợp nhu cầu đã chốt.

Một site hoặc thời gian ngắn vẫn dùng để debug nhưng không được qua gate tuyên bố cross-site/season robustness. Ưu tiên dữ liệu phủ chu kỳ mùa; không đặt số timestamp thay cho số sự kiện/ngày độc lập. Không dừng chỉ vì ESN thua persistence.

### G2 — Feature extractor và kiểm thử vật lý (24 giờ; tuần 2, có thể song song G1)

**Input:** schema dữ liệu, pipeline mẫu, simulator local và circuit specification.

**Việc cần làm:**

- [ ] Bắt đầu Fock input 2 photon, 4–6 mode; lưu V1, V2 và encoding scale để tái lập.
- [ ] Chỉ định backend exact full-distribution bằng `pcvl.BackendFactory.get_backend("SLOS")` (import `perceval as pcvl`), hoặc Processor với backend `"SLOS"`; không dựa vào default. Kiểm tra tên thuộc registry trước khi gọi factory, rồi ghi class/module thực tế và versions Perceval/Exqalibur; tên sai phải fail ở wrapper của dự án, không chấp nhận fallback của thư viện.
- [ ] Khóa phiên bản: tài liệu Perceval v1.2 ánh xạ `"SLOS"` tới `SLOSExqaliburBackend` (C++), còn `"SLOS_LEGACY"` tới `SLOSBackend`. Không coi gọi trực tiếp `SLOSBackend()` luôn tương đương factory `"SLOS"`. SLOS là **Strong Linear Optical Simulation**, lựa chọn phù hợp cho full distribution nhỏ; không tuyên bố nhanh nhất trên mọi workload khi chưa profile.
- [ ] Dùng lớp Simulator/Processor phù hợp cho distinguishability, source/loss và detection; không đưa annotated input thẳng vào backend ideal nếu API không hỗ trợ. Naive chỉ làm oracle đối chiếu ở toy test nhỏ. Finite-shot/noisy path phải khai báo backend/simulator riêng và kiểm toán raw attempts; không coi samples từ phân phối đã postselect là raw source trials.
- [ ] Thiết kế `phi_j = c * s_j(x_j)` với scaler chỉ fit trên train hoặc dùng giới hạn vật lý biết trước. Thử tối đa hai scale ban đầu `c ∈ {π/2, π}` với `s_j ∈ [0,1]`, chọn bằng validation. Đây là grid sàng lọc, không phải ngưỡng vật lý tối ưu đã biết.
- [ ] Khóa chính sách đầu vào ngoài miền: mặc định MinMax theo train với clipping về [0,1], ghi clip flags/tỷ lệ theo feature; không refit trên test. Standardization đơn thuần không bảo đảm pha bị chặn; MinMax không clip cũng không bảo đảm dữ liệu mới nằm trong miền train. Controls nhận cùng preprocessing, đồng thời lưu dữ liệu gốc để kiểm toán mất thông tin.
- [ ] Test miền pha, OOD/clipping, độ nhạy dữ liệu và bão hòa. Đảm bảo encoding tạo pha tương đối quan sát được chứ không chỉ global phase. Scale quá nhỏ cũng có thể làm feature ít biến thiên, khó phân biệt dưới finite shots.
- [ ] Viết test trước cho normalization, photon conservation, global-phase negative control, data sensitivity và HOM control.
- [ ] Thực hiện hai chế độ photon dùng cùng V1/V2 và encoding; kiểm chứng distinguishability đúng.
- [ ] Xây raw-feature và RFF controls với preprocessing/readout nhất quán.
- [ ] Profile exact probabilities và một finite-shot case nhỏ: runtime, peak RAM, kích thước output và dự toán toàn bộ samples × configs × seeds × budgets, gồm noise/loss và chi phí tuning. Không suy ra runtime từ riêng một permanent hoặc một lần suy luận.
- [ ] Hard-cap pilot `n_photons <= 3`; giữ mặc định 2 photon, 4–6 mode. Mọi tăng photon hoặc mode, kể cả 2→3 photon hay 6→8 mode, cần runtime/RAM test giới hạn trước và kiểm tra tổng budget. Vượt 3 photon cần sửa protocol và phê duyệt scope mới; không tự mở sweep.
- [ ] Kiểm tra loại detector/collision filtering, không mặc định full PNR output là chắc chắn có trên QPU tương lai.

**Output:** `artifacts/physics_checks.md`, `artifacts/circuits/`, `artifacts/runtime_profile.csv`; tests có kết quả thật.

**Lưu ý vật lý:** pha tuần hoàn modulo 2π. Encoding có thể làm các input khác nhau trùng pha (aliasing), nhưng không tự biến thành hàm băm hỗn loạn hay bảo đảm mô hình không học được. Giới hạn [0,π/2] hoặc [0,π] giảm rủi ro aliasing do chu kỳ; không bảo đảm output probabilities phân biệt mọi input, vì còn phụ thuộc circuit/measurement. Clipping cũng có thể làm trùng đặc trưng và che mất shift, nên bắt buộc log.

**Lưu ý mô phỏng:** số Fock states ở sector n photon là `D = binomial(m+n-1, n)`. Ví dụ n=3: m=6 có D=56, m=8 có D=120; tăng mode không miễn phí. SLOS tính full distribution có độ phức tạp theo tài liệu là O(nD), nên phải xét cả backend, output và noise model, không chỉ độ phức tạp permanent. Trần 3 photon là rào quản trị pilot, không phải khẳng định 4 photon luôn quá sức CPU.

**Gate:**

- GO: mọi kiểm thử bắt buộc pass, feature thực sự phụ thuộc dữ liệu, runtime của matrix thử nghiệm nằm trong budget.
- EXTEND: một lỗi triển khai cụ thể có hướng sửa; giảm circuit/data size nếu cần profiling, ghi rõ giới hạn.
- STOP cấu hình: không sửa được trong giới hạn công sức hoặc chi phí mô phỏng không phù hợp scope. Lỗi code không phải bằng chứng quantum không có ích.

### G3 — Benchmark có kiểm soát và quyết định pilot (48 giờ; tuần 3–4)

**Input:** dữ liệu qua G1, feature map qua G2, thresholds và test policy đã khóa.

**Việc cần làm:**

- [ ] So raw lags, RFF, distinguishable và indistinguishable với cùng quantile head; XGBoost là đối thủ ứng dụng độc lập; ESN nếu ngân sách cho phép và đã được khai báo trước test.
- [ ] Cân bằng cơ hội tuning; mọi nối raw features vào quantum model cũng phải có control tương ứng.
- [ ] Kiểm tra config trước mỗi run: photon không vượt 3; mode/photon/feature dimension không vượt cấu hình đã profile và duyệt. Có stop condition cho wall-clock/RAM; không mở thêm grid bằng cách ăn vào thời gian evaluation của G3.
- [ ] Dùng các blocked validation origins để chọn cấu hình, không chọn seed đẹp nhất; khởi đầu 5 circuit seeds là các lần lặp riêng, không phải ensemble và không phải bảo đảm statistical power. Báo mean WIS qua 5 lần lặp cùng phân tán giữa seeds; paired block uncertainty trên mean loss theo timestamp chỉ có ý nghĩa điều kiện trên các seeds này. Muốn suy rộng sang phân phối circuit cần thiết kế uncertainty bổ sung.
- [ ] B phụ trách seed manager; A kiểm tra split/RFF/readout. Khóa master seed, seed manifest đa tầng và environment manifest theo quy định dưới đây trước benchmark; không chỉ gọi `np.random.seed()` một lần đầu notebook.
- [ ] Chạy reproducibility smoke test trong hai process mới với cùng manifest; đổi thứ tự jobs vẫn giữ kết quả theo run ID. Gate G3 không qua nếu chưa giải thích được sai khác vượt tolerance đã khóa. Chi phí kiểm tra nằm trong 48 giờ, không tự tăng ngân sách.
- [ ] Freeze lựa chọn và đánh giá trên holdout; không dùng score holdout để tinh chỉnh.
- [ ] Tính proper score chính, calibration, uncertainty bằng paired block bootstrap hoặc phương pháp phù hợp phụ thuộc thời gian; không coi seeds là independent days/sites.
- [ ] Xuất đồng thời quantum-only, baseline trên cùng timestamp thành công, system và baseline trên toàn tập đủ điều kiện; thêm success/fallback rate và nguyên nhân theo chế độ dữ liệu. Không dùng quantum-only có chọn lọc làm tiêu chí GO ứng dụng.
- [ ] Thử một vài raw budgets nhỏ, ví dụ 10³ và 10⁴, để phát hiện feature quá nhạy sampling; ghi accepted counts riêng.
- [ ] Tổng kết GO / EXTEND / STOP / BLOCKED theo mục 5; không bắt buộc phải có kết quả dương.

**Output:** `artifacts/predictions/`, `artifacts/pilot_scores.csv`, `artifacts/resource_ledger.csv`, `reports/pilot_decision.md`.

**Gate:** chỉ cho phép GO sang nghiên cứu finite-shot/loss sâu hơn, không đồng nghĩa GO triển khai thương mại hoặc đã chứng minh quantum advantage.

Primary G3 mặc định là exact-training/exact-test: kiểm tra tiềm năng của feature map. Finite-shot ở đây là screening, không thay thế G4. Ngân sách 5 seed runs phải được cộng trong tổng chi phí thí nghiệm; chi phí mỗi dự báo của một mô hình đơn báo riêng. Nếu sau này dùng ensemble, phải tạo protocol version mới và cộng shots của tất cả circuit cho mỗi dự báo.

#### Seed contract và bằng chứng tái lập G2–G4

- Tạo các stream riêng bằng `numpy.random.SeedSequence` từ master seed và bộ ID số nguyên ổn định (component, circuit replicate, sampling replicate, input ID khi cần). Ghi rõ mapping/algorithm; không dùng `hash()` mặc định của Python hoặc thứ tự worker hoàn thành để sinh seed. Lưu seed số nguyên tương thích API, master entropy và ID trong `artifacts/seed_manifest.json`.
- NumPy: truyền `default_rng(child_seed)` tường minh cho code sở hữu RNG. Legacy code dùng global RNG phải được seed có kiểm soát. scikit-learn không có một global seed riêng: truyền integer `random_state` cho RBFSampler và estimator/search ngẫu nhiên; khóa seed cho XGBoost/readout nếu có ngẫu nhiên. Không để `None` tại các thành phần ngẫu nhiên đã dùng.
- Split train/validation/test là chronological manifest cố định, không shuffle và không tạo lại theo circuit seed. `split_seed` chỉ cần khi có bước chọn khối/site ngẫu nhiên được đăng ký riêng, không thay thế chính sách chống leakage. Bootstrap có stream riêng, không thay đổi circuit hay split.
- Perceval: gọi `pcvl.random_seed(seed_circuit)` ngay trước khối tạo V1/V2 ngẫu nhiên, lưu cả ma trận/tham số và checksum để tái dựng mà không cần RNG. Theo implementation đã đối chiếu, hàm này seed Python `random`, NumPy legacy RNG và Exqalibur; nó không seed lại các `default_rng` độc lập. Không giả định chỉ seed NumPy sẽ bao phủ C++ RNG.
- `pcvl.random_seed` thay đổi global state: cô lập công việc dùng RNG global trong process riêng hoặc chạy tuần tự, không cho nhiều threads đồng thời reseed. Với finite-shot/noise, dùng stream sampling riêng cho từng job/input theo manifest, không reset cùng một seed cho mọi timestamp; kiểm thử API thực tế của phiên bản đã khóa.
- Năm circuit seeds ở primary exact G3 chỉ thay V1/V2; giữ split, RFF, readout và tuning randomness cố định để không trộn variance của nhiều nguồn rồi gọi là circuit variance. Hai đối chứng distinguishable/indistinguishable dùng đúng cùng V1/V2 đã lưu. Nếu khảo sát thêm RFF/readout/sampling seeds, đó là trục thí nghiệm riêng có budget và báo cáo riêng, không tự mở tích Descartes tất cả seeds.
- Lưu `artifacts/environment_manifest.json`: Python, NumPy, scikit-learn, XGBoost, Perceval, Exqalibur, backend class/module, OS/CPU, BLAS/thread settings, code/config/data hashes và thứ tự feature/output states chuẩn hóa. Seed không bảo đảm bitwise identity giữa các phiên bản, kiến trúc hay thuật toán không deterministic.
- Nghiệm thu: hai process cùng manifest tái lập split IDs, V1/V2 và RFF weights; exact features/predictions/WIS khớp trong tolerance chốt trước. Sampling counts phải lặp lại trong cùng môi trường/chế độ deterministic đã xác minh; đổi sampling seed không thay circuit/split, đổi circuit seed không thay split/RFF. Kiểm tra danh tính backend và so phân phối SLOS với Naive trên toy case bằng tolerance. Khi đổi môi trường phải revalidate, không mặc định tái lập tuyệt đối.
- Output bổ sung: `artifacts/seed_manifest.json`, `artifacts/environment_manifest.json`, `artifacts/reproducibility_report.md`. Nếu thất bại, đánh dấu invalid/EXTEND sửa kỹ thuật trong budget, không diễn giải thành bằng chứng khoa học âm.

### G4 — Finite-shot, loss và shift (chỉ sau phê duyệt; 4–6 tuần bổ sung)

**Input:** pilot có lý do tiếp tục rõ, ngân sách mới và miền dữ liệu/test mới khi cần.

**Việc cần làm:**

- [ ] So postselection, direct lossy features và recycling mitigation trên cùng raw measurement budget.
- [ ] Tách train/test exact và finite; log nguồn loss, source/detector model và postselection.
- [ ] Giữ cách tách quantum-only/system của G3, kể cả khi accepted counts thấp do loss. Khóa success rule trước test, không loại dự báo theo sai số sau khi biết nhãn; cộng cả chi phí quantum thất bại và chi phí fallback.
- [ ] Kiểm tra weather shift độc lập với hardware mismatch; phân tích theo site/mùa nếu dữ liệu hỗ trợ.
- [ ] Quét tuần tự budgets 10²–10⁴; chỉ thêm 10⁵ như diagnostic khi được duyệt, không ngầm coi là triển khai khả thi.
- [ ] Xác định vùng accuracy–calibration–cost hợp lý; xem lợi ích còn tồn tại so với B và physical controls không.

**Output:** `reports/robustness_budget.md`, đường trade-off, cấu hình được/không được khuyến nghị và yêu cầu phần cứng.

**Gate:** GO tới kiểm chứng phần cứng nếu có cấu hình đạt hiệu quả và budget đề xuất; EXTEND nếu thiếu một yếu tố có thể giải quyết; STOP use case nếu lợi ích mất trong mọi cấu hình của phạm vi đã chốt. Đây không phải kết luận trên mọi QRC.

Primary G4 đề xuất là finite-training/finite-test, 10⁴ raw attempts cho mỗi input/circuit, với source-trial semantics và loss scenario chính được khóa trước test. Các budgets khác là secondary. Có thể chọn mức khác ở G0 của giai đoạn này nếu profiling/nguồn lực yêu cầu, nhưng không đổi primary sau khi xem holdout. Không dùng chiến thắng exact của G3 để vượt qua gate đo hữu hạn G4.

### G5 — QPU và thẩm định ứng dụng (chưa lập lịch)

**Input bắt buộc:** access/quota thật; đặc tả detector, raw events, calibration và API; quyền công bố; chi phí và latency SLA được chốt.

**Việc cần làm:** kiểm tra sim-to-hardware, đo E2E thật, tính chi phí mỗi dự báo, chạy shadow evaluation không tác động vận hành.

**Output:** báo cáo hardware, cost/SLA và khuyến nghị đầu tư tiếp hoặc dừng.

**Gate:** chưa có access → BLOCKED ở G5, không biến thành kết quả âm và không lấy thời gian simulator/readout thay E2E. Nếu hardware không đạt SLA hoặc chi phí vượt ngưỡng và không có cách sửa trong budget → STOP triển khai này. Paper và sản phẩm phải được duyệt bằng tiêu chí riêng.

## 5. Tiêu chí định lượng và quyết định

### Metric

- Primary: WIS trên daylight, cùng quantile grid, chuẩn hóa theo capacity khi so nhiều site. Chốt cách gộp site/khối trước test.
- Skill ứng dụng so baseline B: `S_sys = 1 - WIS_system(T) / WIS_B(T)`. Trong bảng quyết định bên dưới, `S` nghĩa là `S_sys`. S > 0 là toàn hệ thống tốt hơn B trên cùng tập đánh giá. Chỉ dùng khi mẫu số khác 0; nếu baseline score bằng 0, không có headroom để cải thiện score đó.
- δ = 0,02 (2%) là **đề xuất** để thảo luận, phải được duyệt trước test.
- Coverage của khoảng 90%, độ rộng khoảng và WIS cùng được báo; không cải thiện coverage bằng cách mở khoảng vô hạn.
- Raw attempts, accepted counts, number of circuits, training/tuning và feature extraction costs phải cộng đầy đủ.

### Tách metric quantum-only và system: bắt buộc

`T` là tập timestamp daylight đủ điều kiện đánh giá theo QC/split/target policy của G1, độc lập với việc quantum thành công. `A ⊆ T` là các timestamp quantum thành công theo điều kiện đã khóa (input hợp lệ, accepted counts đủ, output hợp lệ và trong timeout/budget), xác định không dùng nhãn tương lai hoặc prediction error.

| Báo cáo | Timestamp | Score mô hình | Baseline đối chiếu |
|---|---|---|---|
| Quantum-only (có điều kiện) | A | WIS_Q(A) | WIS_B(A) |
| System — metric ứng dụng chính | T | WIS_system(T): quantum trên A, fallback trên T\\A | WIS_B(T) |

- Báo `S_Q = 1 - WIS_Q(A)/WIS_B(A)` riêng, cùng success rate `|A|/|T|`, fallback rate và số mẫu; nếu A rỗng, quantum-only là NA, không phải 0. Quy tắc mẫu số bằng 0 áp dụng như trên.
- Baseline trên cùng A bảo đảm so sánh cùng timestamp, nhưng không xóa selection bias: A có thể thiếu các trường hợp khó. Không suy rộng quantum-only sang T; phân tích missing/fallback theo thời tiết/ramp khi metadata cho phép.
- Báo WIS, coverage, interval width và uncertainty theo từng tập. Bootstrap các khối thời gian gốc, giữ success/failure mask; không nén A thành chuỗi liên tục giả. Với nhiều seeds, báo matched-baseline score và success rate theo seed trước khi gộp theo protocol.
- Đối chứng vật lý phải dùng cùng tập timestamp cho mọi nhánh được so, hoặc báo giao các tập thành công và tỷ lệ bao phủ của giao đó; không so mỗi nhánh trên tập thuận lợi riêng.
- Resource ledger của system gồm quantum attempts thất bại, thời gian timeout và fallback. Nếu cả fallback cũng không tạo được dự báo, đó là unresolved system failure: không âm thầm xóa timestamp để báo WIS toàn T; báo score phần quan sát được và đánh giá toàn tập chưa hoàn tất.
- G3 exact có thể có A=T khi không lỗi; G4 finite-shot phải kiểm tra lại. Chỉ quantum-only đẹp không đủ GO. Tuyên bố đóng góp lượng tử còn cần controls, không dựa riêng vào System Score.

### Quy tắc quyết định chính cho nhánh ứng dụng

| Tình trạng | Quyết định | Điều kiện/giới hạn |
|---|---|---|
| Cận dưới CI 95% của S > δ, calibration và resource gates đạt | GO | Qua giai đoạn kế tiếp, không suy ra advantage tổng quát |
| CI còn bao gồm cả kết quả dưới và trên δ | EXTEND có điều kiện | Chỉ nếu có thí nghiệm cụ thể khả thi trong 24 giờ bổ sung để giảm bất định |
| Cận trên CI < δ và không có lợi ích chi phí/nghiệp vụ khác đã đăng ký | STOP nhánh ứng dụng theo mục tiêu hiện tại | Có thể có cải thiện nhỏ, nhưng không đạt ngưỡng giá trị đã chốt |
| S đạt nhưng calibration/resource không đạt | EXTEND hoặc STOP | Một can thiệp có lý do, giới hạn budget; không mở tuning vô hạn |
| Chỉ thắng physical control nhưng không đạt so với B | Không GO ứng dụng | Đề nghị nhánh khoa học riêng chỉ khi có insight, novelty và được Quang duyệt |
| Thắng B nhưng ngang distinguishable | Không tuyên bố cần giao thoa nhiều photon | Kiểm tra giải pháp cổ điển rẻ hơn; không tự cấp tiền cho QPU |
| Dữ liệu, quyền truy cập hoặc code chưa đủ để đánh giá | BLOCKED/invalid | Báo nguyên nhân; không gán nhãn kết quả khoa học dương/âm |

CI là công cụ ra quyết định có giả định, không phải chứng minh tuyệt đối. Bootstrap phải giữ cấu trúc thời gian; báo cách chọn block và kiểm tra sensitivity. Primary claim được chốt trước; nếu xác nhận nhiều claims/đối chứng thì phải có kế hoạch multiple-comparison hoặc kiểm định phân cấp. Chưa đủ dữ liệu cho interval đáng tin cậy → feasibility/inconclusive, không giả tạo CI hẹp.

## 6. Chính sách thử nghiệm bổ sung: hữu hạn, có lý do

Mỗi đề nghị EXTEND phải ghi:

1. Điều chưa biết là gì? Ví dụ variance theo circuit seed, data quality, sensitivity với loss.
2. Thay đổi duy nhất hoặc nhóm thay đổi có kiểm soát cần thử.
3. Kết quả nào sẽ khiến tiếp tục, kết quả nào sẽ khiến dừng.
4. Giờ người/compute tối đa, người phụ trách và ngày quyết định.
5. Tập dữ liệu nào chưa dùng cho tuning sẽ đánh giá lần mới.

Mặc định tối đa **một đợt bổ sung 24 giờ kỹ sư**, ngoài 96 giờ pilot và chỉ khi được duyệt. Nếu sau đợt này vẫn inconclusive: kết thúc pilot với kết luận chưa đủ bằng chứng; không tự gia hạn. Việc thu thêm dữ liệu theo mùa có thể cần thời gian lịch dài hơn và phải là đề nghị mới.

Khi đã xem test rồi thay mô hình, test cũ trở thành dữ liệu phát triển. Đánh giá xác nhận lần sau cần holdout mới hoặc giao thức sequential đã định trước; không lặp mở cùng test và giữ CI 95% như chưa có lựa chọn thích nghi.

## 7. File/interface đề xuất cho bước triển khai sau

Các file dưới đây **chưa được tạo hoặc thực thi**. Khi được yêu cầu code, rà repository thực tế rồi lập task-level plan/TDD theo các interface này.

| File/module dự kiến | Trách nhiệm | Kiểm thử nghiệm thu |
|---|---|---|
| `configs/pilot.yaml` | Target, grids, thresholds, budgets, seeds, success/fallback policy | Config đầy đủ; reject photon >3 và cấu hình chưa profile; không silent defaults làm đổi protocol |
| `src/pv_pilot/data.py` | Schema, timestamps, target, availability, QC manifest | `tests/test_data.py`: alignment/leakage/missing policy; không nối cửa sổ qua gap |
| `src/pv_pilot/splits.py` | Chronological manifests | `tests/test_splits.py`: label cutoff, holdout không vào fit |
| `src/pv_pilot/reproducibility.py` | Seed hierarchy, stable run IDs, manifests, process isolation | `tests/test_reproducibility.py`: hai process, thứ tự jobs, tách circuit/sampling/RFF streams |
| `src/pv_pilot/classical.py` | Persistence/RFF/XGBoost và controls | `tests/test_classical.py`: baseline trên chuỗi synthetic |
| `src/pv_pilot/photonics.py` | V1–D–V2, phase scaling, backend tường minh, physics modes/detection | `tests/test_photonics.py`: HOM/global phase/normalization; phase range, clipping/OOD, sensitivity; backend identity và toy SLOS–Naive equivalence |
| `src/pv_pilot/readout.py` | Quantile loss, head và ordering policy | `tests/test_readout.py`: cùng scalar labels; quantiles hợp lệ |
| `src/pv_pilot/evaluation.py` | Quantum-only/system WIS, matched baselines, coverage, paired uncertainty | `tests/test_evaluation.py`: toy examples; tập A/T, A rỗng, fallback lỗi, baseline cùng timestamp |
| `src/pv_pilot/resources.py` | Raw/accepted samples, runtime/RAM và budget ledger | `tests/test_resources.py`: không bỏ shots/filter/ensembles/failed attempts/fallback |
| `reports/pilot_decision.md` | Quyết định và giới hạn bằng chứng | Mọi kết luận nối tới run/config/prediction đã lưu |

TDD ở bước code: viết kiểm thử nhỏ → xác nhận fail đúng nguyên nhân → triển khai tối thiểu → test pass → kiểm tra hồi quy. Lệnh dự kiến `python -m pytest tests/`; chưa có kết quả test nào để báo pass trong bản kế hoạch này.

## 8. Mẫu decision memo một trang

- **Gate/ngày/người phê duyệt:**
- **Budget đã dùng/còn lại:**
- **Dataset, protocol và model versions:**
- **Baseline B được chọn thế nào:**
- **Quantum-only WIS/skill + baseline trên A; system WIS/skill + baseline trên T; CI tương ứng:**
- **Success/fallback rate, nguyên nhân thất bại; calibration; tài nguyên gồm failed attempts:**
- **Kết quả physical controls:**
- **Điều đã chứng minh / điều chưa thể kết luận:**
- **GO / EXTEND / STOP / BLOCKED và lý do:**
- **Nếu EXTEND:** một câu hỏi, một thí nghiệm, trần công sức và holdout mới.
- **Nếu STOP:** đóng phạm vi nào; lưu code/data manifests/negative results; điều kiện nào cho phép mở lại.

## 9. Tài liệu nền

- [Đánh giá thiết kế và literature đã đối chiếu](../../../Photonic_PV_pilot_review_20260926.md).
- [Danh mục nguồn, trạng thái peer review và giới hạn đọc](../../../sources/Photonic_PV_pilot_20260926_references.md).
- [Perceval: SLOS backend và độ phức tạp full-distribution](https://perceval.quandela.net/docs/v1.0/reference/backends/slos.html) — nguồn cho cảnh báo tài nguyên, không phải benchmark runtime của pilot.
- [scikit-learn: MinMaxScaler và chính sách clip](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) — miền train không tự bảo đảm miền dữ liệu mới.
- [Perceval v1.2: SLOS và SLOS_LEGACY](https://perceval.quandela.net/docs/v1.2/reference/backends/slos.html); [factory source](https://github.com/Quandela/Perceval/blob/main/perceval/backends/__init__.py) — tên viết hoa, default và fallback phải kiểm tra theo version khóa, không coi nhánh main là dependency pin.
- [Perceval random_seed implementation](https://github.com/Quandela/Perceval/blob/main/perceval/utils/_random.py); [NumPy SeedSequence/parallel streams](https://numpy.org/doc/stable/reference/random/parallel.html); [RBFSampler random_state](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html) — cơ sở cho seed contract, chưa thay thế test local.

Kế hoạch kế thừa bằng chứng của báo cáo trên, không bổ sung tuyên bố SOTA mới. Kỹ năng lập kế hoạch được áp dụng ở mức khung theo yêu cầu; kỹ năng đánh giá khoa học dẫn đến việc thêm trạng thái BLOCKED/inconclusive, giới hạn EXTEND và chống test reuse. Không có hình kết quả hoặc benchmark giả.
