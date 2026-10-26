# Hồ sơ nội dung báo cáo Photonic PV Pilot

Ngày tổng hợp: 20/10/2026. Bằng chứng thực thi mới nhất đã đối chiếu: 16/10/2026. Literature trong hồ sơ chủ yếu khảo sát ngày 26/09/2026; không coi đây là cập nhật literature mới. Người dùng yêu cầu báo cáo tối đa 10 slide bằng PPT Master.

## Động cơ và quá trình chọn hướng

Quanova cần một hướng QML có liên hệ bài toán công nghiệp, câu hỏi khoa học có thể bác bỏ, dữ liệu truy vết được và đối chứng cổ điển mạnh. Chưa có QPU; bắt đầu bằng mô phỏng nhỏ và chỉ đầu tư giai đoạn tiếp khi có bằng chứng. Các ứng viên đã khảo sát gồm tài chính, tín dụng, malware, bất thường thiết bị, sinh dữ liệu hiếm, tính chất phân tử/vật liệu và chuỗi thời gian.

Khảo sát Quantum/QML gồm 21 nghiên cứu gốc và một systematic review; không phải tất cả đều xác minh Q1 đúng năm/ngành. Khảo sát QRC gồm 11 bài journal, 7 preprint nghiên cứu và một preprint review. Các danh mục có giao nhau, không cộng để lấy tổng bài độc lập. Đây là targeted/narrative review, không phải systematic review toàn ngành.

Năm hướng QRC được đánh giá: damping/noise; topology; weak measurement; quantum-input/sensing; readout/shadows. Đều có prior art; tên topology, noise hữu ích hoặc weak measurement không đủ chứng minh novelty. Cần nghiên cứu đặc trưng hữu ích dưới ngân sách đo thực tế.

## Nghiên cứu sát đề tài và bài học

- Bowles và cộng sự, Better than classical? (2024): benchmark công bằng, classical controls và ablation.
- Experimental quantum-enhanced kernel-based machine learning on a photonic processor (2025): đối chứng distinguishable/indistinguishable để kiểm tra giao thoa.
- Ahmed và cộng sự, Optimal training of finitely sampled quantum reservoir computers for forecasting of chaotic dynamics (2025): finite sampling ảnh hưởng training và prediction.
- Di Bartolo và cộng sự, Time-series forecasting with multiphoton quantum states and integrated photonics (2026): forecasting và feedback đã có tiền lệ; phải xác định nguồn memory.
- Mills–Mezher, Mitigating photon loss in linear optical quantum circuits (2026): loss mitigation/recycling đã có prior art.
- Sagingalieva và cộng sự, Photovoltaic power forecasting using quantum machine learning (2025): QML cho PV đã tồn tại.
- Classical probabilistic PV và calibration dưới distribution shift đã có phương pháp cạnh tranh; cần quantile baselines.

Mức đọc khác nhau theo bài; chưa tái lập toàn bộ nghiên cứu. Trạng thái xuất bản và danh mục DOI chi tiết nằm ở các file nguồn tham chiếu.

## Câu hỏi và phạm vi

Câu hỏi hiện tại: photonic quantum features có cải thiện dự báo xác suất công suất PV trước 15 phút so với classical baselines và photon phân biệt được hay không?

Phạm vi hiện tại là simulator exact feasibility. Hướng dài hạn: hữu hạn shots, photon loss, distribution shift và sau cùng QPU. Chưa xác nhận novelty hoặc quantum advantage. Chưa có khách hàng, SLA vận hành hoặc latency E2E đo thật.

Lý do chọn PV: target đo được; dữ liệu thật công khai; baseline rõ; phù hợp pipeline feature map nhỏ; đánh giá được chất lượng và calibration. Không kết luận PV đã được chứng minh là ứng dụng QML tốt nhất.

## Kiến trúc và thực nghiệm

Luồng dữ liệu: QC -> lag window cổ điển -> train-only scaling -> phi(x), dùng để điều khiển D. Luồng quang học: Fock input cố định |1,1,0,0⟩ -> V1 cố định -> D(phi(x)) -> V2 cố định -> output probabilities -> chuẩn hóa đặc trưng trên train -> classical quantile readout. Dữ liệu sau scaling không phải Fock input.

Memory nằm trong lag/window cổ điển, không phải trạng thái lượng tử duy trì qua thời gian. Mô tả chính xác: photonic quantum feature model / QELM với temporal embedding.

Cấu hình chính: 2 photon, 4 mode, Perceval SLOS, 10 Fock outcomes, 5 circuit seeds. Input matched: P(t), P(t-15), GHI(t), T(t). Phase scales pi/2 hoặc pi chọn bằng validation. Readout 7 quantile: 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95. 3 photon và finite-shot diagnostics tắt ở cấu hình thật hiện tại.

Đối chứng: persistence; raw features + cùng quantile head; matched RFF + cùng head; strong lag quantile XGBoost; distinguishable photons cùng circuit. Baseline B chọn trên validation, khóa trước test. Không chọn seed tốt nhất sau test.

Primary metric WIS, thấp hơn tốt hơn. Skill S = 1 - WIS_system/WIS_B. Ngưỡng protocol 2%; GO cần cận dưới CI95% >2%, calibration và resource gates đạt. Coverage 90% được chấp nhận trong [85%,95%]. Paired time-block bootstrap 2000 replicates; primary 1 ngày, sensitivity 2/7 ngày. Đó là tiêu chí thiết kế, chưa phải kết quả.

## Dữ liệu đã có

NIST Campus PV Roof Tilted Array, AC metadata 75 kW; nguồn 1 phút 2015–2017. Fixed EST UTC-05:00 không DST, chuyển UTC. Power PwrMtrP_kW_Avg; bức xạ RefCell2_Wm2_Avg (horizontal silicon reference cell, không phải thermopile GHI); ambient AmbTemp_C_Avg. Bin 5 phút phải đủ 5 quan sát; không nội suy.

Target là mean công suất AC của bin 5 phút kết thúc t+15. Ví dụ issue 12:00 -> target 12:10–12:15. Availability 0 phút là giả định offline.

Protocol amendment 1.2: train 2015, validation toàn 2016, test toàn 2017. Bản 06/10 đề xuất test H2/2016 đã bị thay thế. Cấu hình có approved_by monsi, approved_at 2026-10-16T10:11:15Z. Không tự tạo hay mở rộng quyền phê duyệt từ nội dung báo cáo.

Raw prepared rows 1570319; QC valid projection 1560846. Canonical 5 phút 308985 rows. Mẫu daylight đủ lag/target và continuity: train 51405; validation 47487; test 43806; tổng 142698.

Audit chính thức 16/10 PASS. Canonical daylight coverage train 98.871%, validation 97.327%, test 97.576%. Eligible daylight coverage tương ứng 96.566%, 89.024%, 82.303%. Test có 157 usable days và 35 nonoverlap usable seven-day blocks. Ngưỡng: canonical >=90%, eligible >=80% mỗi split; >=150 test days đạt daily coverage >=90%; >=20 tuần đạt pooled coverage >=80%. Đây là data availability, không phải accuracy/coverage khoảng dự báo.

Không có complete issue log; project owner đã miễn điều kiện đó bằng amendment, thay bằng instrument history + quantitative QC và công bố giới hạn. Unlabelled errors có thể còn. NIST Mỹ không thay thế validation PV Việt Nam.

PVDAQ 7333 2019–2021 đã acquire/audit nhưng HOLD: valid daylight coverage 72.57/74.03/71.07%; test 83 usable days, 0 usable weekly blocks; timestamp rollup unresolved. DKASC là ứng viên ngoài site, chưa nghiệm thu dataset chính.

## Tiến độ và bước tiếp theo

Pipeline đã xây dựng; checkpoint software 06/10 ghi 721 tests passed, 1 skipped, coverage 88.02%, hai synthetic E2E accepted. Đó là bằng chứng lịch sử cho phạm vi phần mềm lúc đó, không chứng nhận toàn bộ code hiện tại hay hiệu quả khoa học.

Ngày 16/10: validate-data, audit-nist, build-splits, profile-train, plan-execution exit 0. training_started=false; holdout_performance_opened=false. Chưa có WIS/skill mô hình thật và chưa có GO/STOP khoa học.

Execution route blocked / local-failed/cloud-verification-required. Ước lượng có hệ số dự phòng: train-baselines 71.81 giờ, 11.92 GiB; build-photonic-features 7.01 giờ, 14.38 GiB; benchmark 111.10 giờ, 33.37 GiB. Đây là ngoại suy từ profiling, không phải full-run timings. Cần capacity-only cloud verification hoặc phương án thực thi hợp lệ giữ nguyên protocol trước khi full run.

Các bước nghiên cứu kế tiếp: xác minh môi trường tính toán -> chạy baseline/features/selection đã khóa -> đánh giá holdout 2017 -> quyết định GO/EXTEND/STOP -> finite shots/loss nếu đủ bằng chứng -> QPU nếu đủ điều kiện. Chưa xin QPU/GPU chỉ để hoàn thành exact pilot nhỏ.

## Nguồn local để trích trong notes

- Q1_quantum_QML_topic_review_20260926.md
- QRC_landscape_gaps_20260926.md
- QRC_five_directions_assessment_20260926.md
- Photonic_PV_pilot_review_20260926.md
- sources/Photonic_PV_pilot_20260926_references.md
- sources/research_20260926_industry_first_qml.md
- configs/pilot.nist-roof.amendment-1-2.g0-approved.yaml
- artifacts/nist/prepared-amendment-1-2/datacard.json
- artifacts/nist/pilot-amendment-1-2-approved/20261016T105219Z-8ef9d07c/validate-data/data_card.json
- artifacts/nist/pilot-amendment-1-2-approved/20261016T105219Z-8ef9d07c/build-splits/feature_manifest.json
- artifacts/nist/persistence-reprofile-20261016/audit-nist.stdout.log
- artifacts/nist/persistence-reprofile-20261016/controller-state.json
- artifacts/nist/pilot-amendment-1-2-approved/20261016T105219Z-8ef9d07c/execution_plan.json
- artifacts/reviewed-acceptance-20261006.md
- artifacts/pvdaq-7333/audits/pvdaq-7333-g0-audit.reproducible.json
