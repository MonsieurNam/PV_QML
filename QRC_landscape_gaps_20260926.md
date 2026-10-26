# QRC: nghiên cứu quốc tế và khoảng trống ứng viên cho Quanova

Ngày khảo sát: 26/09/2026. Tổng hợp có chọn lọc, không phải systematic review. Danh mục gồm 11 bài journal, 7 preprint nghiên cứu và 1 preprint review; trạng thái xuất bản được tách rõ. Không coi đây là danh sách Q1 đã chứng nhận. ID J/P bên dưới trỏ đến [danh mục và search log](sources/qrc_20260926_references.md).

## 1. Kết luận điều hành

QRC đáng khảo sát vì phần reservoir có thể giữ cố định và chỉ học readout cổ điển. Nhưng tránh gradient mạch không đồng nghĩa tránh chi phí đo, model selection hoặc giới hạn thống kê. Hướng thực tế cho Quanova là nghiên cứu **đặc trưng có thể đo và sử dụng được trong ngân sách hữu hạn**, với một bài toán cụ thể và đối chứng mạnh; chưa nên hứa quantum advantage.

Đề xuất ưu tiên: **đánh đổi giữa bậc tương quan, độ chính xác dự đoán và ngân sách lấy mẫu trong photonic reservoir**. Chốt một trong hai loại task: temporal QRC cho tín hiệu công nghiệp, hoặc static reservoir/QELM cho tính chất vật liệu. Không trộn hai loại để tạo cảm giác cùng một cơ chế.

## 2. Khung khái niệm

Với temporal QRC, một mô tả đơn giản là:

`u_t → encoding → reservoir có trạng thái/bộ nhớ → đo features z_t → readout cổ điển → dự đoán`

Trong mô hình lý tưởng không có feedback điều kiện phức tạp: `rho_(t+1) = E_(u_t)(rho_t)`, `z_(t,j) = Tr(O_j rho_t)`, `y_hat_t = w^T z_t + b`. Trên máy thật chỉ có ước lượng hữu hạn của z. Feedback dựa trên kết quả đo cần đưa cả trạng thái cổ điển vào mô hình. Nền tảng QRC: [J1](https://doi.org/10.1103/PhysRevApplied.8.024030).

| Loại | Bộ nhớ nằm ở đâu? | Task tự nhiên | Cảnh báo |
|---|---|---|---|
| Stateful QRC | Trạng thái vật lý duy trì qua thời gian | Chuỗi thời gian | Đo có thể thay đổi trạng thái; không giả định đọc vô hạn không tốn phí |
| Measurement-feedback / hybrid QRC | Kết quả đo được lưu và hồi tiếp, có thể kết hợp memory vật lý | Chuỗi thời gian | Phải đối chứng cùng memory cổ điển |
| Stateless reservoir / QELM | Không lưu giữa mẫu độc lập | Molecular/material regression, classification | Không tự có lợi thế xử lý temporal |
| QRP cho quantum data | Đầu vào là trạng thái lượng tử | Tomography, property estimation | Baseline phải là phương pháp đo/suy luận thích hợp |

Các ranh giới này được minh họa bởi J2/J3/J8 và [P3](https://arxiv.org/abs/2602.14677), không chỉ bởi cách tác giả đặt tên.

## 3. Những hướng quốc tế đang theo đuổi

### 3.1 Memory, online operation và measurement back-action

[J2](https://www.nature.com/articles/s41534-023-00682-z) nghiên cứu weak/projective measurement và đánh đổi giữa lấy thông tin, độ nhiễu và bộ nhớ. [J3](https://www.nature.com/articles/s41467-024-51162-7), Princeton–IBM–RTX BBN, kết hợp lý thuyết và triển khai mid-circuit measurement/reset cho xử lý temporal vượt khoảng coherence đơn lẻ. [J10](https://www.nature.com/articles/s41534-026-01229-8) khảo sát controlled damping. [P6](https://arxiv.org/abs/2607.14311), preprint tháng 7/2026, đưa framework chung cho các kiểu monitoring.

Hàm ý: “thêm bộ nhớ”, “noise có ích” hoặc “dùng weak measurement” không còn là gap nguyên vẹn. Cần xác định cơ chế và lợi ích tài nguyên trong một điều kiện cụ thể.

### 3.2 Photonic QRC có memory thực nghiệm

[J7](https://www.nature.com/articles/s41566-026-01880-9), Nature Photonics 2026, nhóm LKB–IFISC, dùng squeezed states, homodyne và feedback; mở rộng độ trễ có spatial multiplexing. [J8](https://www.nature.com/articles/s41534-026-01236-9), npj Quantum Information 2026, nhóm Sapienza–CNR, khảo sát multiphoton và feedback; đối chứng photon phân biệt/không phân biệt. Bộ nhớ không nên được quy hoàn toàn cho quantum correlations.

Hai nền tảng khác nhau: continuous-variable squeezed light không chuyển nguyên xi sang nguồn single-photon Fock và interferometer của Quandela.

### 3.3 Scaling và quantum correlations

[J9](https://www.nature.com/articles/s41534-026-01251-w), Cornell/NTT/Sapienza, dùng GBS trên 400 mode cho classification; so sánh readout trung bình và tương quan, cùng các nguồn sáng. [P1](https://arxiv.org/abs/2407.02553), QuEra và cộng tác viên, báo cáo reservoir learning đến 108 qubit. P1 được giữ ở nhãn preprint trong khảo sát này. Số mode/qubit không chứng minh end-to-end advantage; task advantage dựng từ kernel geometry cũng không phải ứng dụng công nghiệp phổ quát.

### 3.4 Đọc ra thông tin hữu ích với ít phép đo

[J4](https://link.springer.com/article/10.1007/s42484-025-00261-9) nghiên cứu finite-sampling noise và denoising bằng SVD/filtering. [P3](https://arxiv.org/abs/2602.14677) tối ưu observable bằng kernel ridge framework, có thảo luận triển khai. [P2](https://arxiv.org/abs/2505.10062) phân tích concentration trong scrambling reservoirs và vai trò đối xứng. P2/P3 là preprint trong hồ sơ tìm được.

Hàm ý: “lọc noise bằng PCA/SVD”, “chọn observable tốt hơn”, “dùng symmetry” đã có prior art. Gap ứng viên phải đặt thêm ràng buộc thực thi cụ thể và chứng minh khác biệt với các phương pháp gần nhất.

### 3.5 Ứng dụng khoa học và công nghiệp

[J5](https://pubmed.ncbi.nlm.nih.gov/40764018/) nghiên cứu molecular activity ở chế độ ít mẫu; có code chính thức, nhưng reservoir neutral-atom không phải photonic. [J11](https://pubmed.ncbi.nlm.nih.gov/41964986/), PRL 2026, báo cáo thí nghiệm 9-spin và dự báo thời tiết tốt hơn các classical reservoirs được khảo sát. Không diễn giải thành thắng mọi phương pháp forecasting hoặc nhanh hơn khi tính toàn bộ phép đo. [P7](https://arxiv.org/abs/2602.14641) khảo sát neutral-atom trên dữ liệu y sinh nhỏ, phân biệt simulation/hardware và hiệu ứng regularization; không phải bằng chứng triển khai lâm sàng.

### 3.6 Quantum-native tasks và Quandela

[J6](https://pmc.ncbi.nlm.nih.gov/articles/PMC12700208/), Science Advances 2025, dùng QELM cho entanglement witnessing, với đối chứng đo trên cùng thiết bị và thử suy rộng từ separable sang entangled states. [P4](https://arxiv.org/abs/2605.10471), Quandela–Warsaw, triển khai photonic reservoir cho tomography và classical classification; có hardware-aware perturbation khi huấn luyện. P4 vẫn được ghi là preprint. Một basis đo không có nghĩa tomography tổng quát có chi phí hằng số: số outcomes, số mẫu và postprocessing vẫn phải được tính.

### 3.7 Phản biện advantage và baseline

[P5](https://arxiv.org/abs/2607.09905), preprint 2026, đưa hai kiểm tra nhỏ tới 11 qubit nơi classical controls xóa lợi ích được kỳ vọng. Giới hạn: chỉ hai cấu hình; một task có quy luật quadratic thuận lợi cho polynomial baseline, và số features không bằng nhau tuyệt đối. Không dùng bài này để kết luận toàn QRC không có giá trị. Nó là lý do bổ sung các đối chứng cơ chế, không phải phán quyết cuối cùng.

## 4. Bốn gap ứng viên, không phải bốn novelty đã được chứng minh

### A — Readout và tương quan dưới ngân sách lấy mẫu thực tế (ưu tiên 1)

**Câu hỏi:** Khi tăng số photon/mode và bậc tương quan, ở mức ngân sách nào features bổ sung giúp dự đoán, và từ mức nào variance/loss/conditioning làm lợi ích biến mất?

**Prior art bắt buộc:** J4, J9, P2, P3; feature selection cũng đã có ở các nghiên cứu khác, bao gồm P7. Không tuyên bố là người đầu tiên tối ưu readout.

**Phạm vi ứng viên cụ thể:** photonic Fock reservoir, photon-count data, lựa chọn/regularization readout theo noise covariance; tìm quy tắc dự đoán crossover, không chỉ sweep accuracy. Đặt cạnh SVD, ridge và lựa chọn features chuẩn; kiểm tra trên một họ task cơ chế và một task có ý nghĩa ngành.

**Điểm dễ sai:** nhiều moments/correlators có thể lấy từ cùng record. Bỏ features không tự động giảm số lần chạy QPU. Phải đo số shots tối thiểu để đạt mức lỗi cố định, hoặc giảm measurement settings/latency/postprocessing thực sự.

Một phép tính kiểm tra, không phải novelty: nếu `N ~ Multinomial(S,p)`, `p_hat=N/S` và readout `w` cố định, thì

`Var(w^T p_hat) = [sum_j w_j^2 p_j - (sum_j w_j p_j)^2] / S`.

Covariance giữa bins không bằng không. Công thức chưa tính uncertainty do fit w, drift và postselection. Nếu đã dùng nhiều basis/setting, phải nhóm và tính ngân sách tương ứng.

**Kết quả có giá trị:** một estimator/selection rule hoặc bound giải thích trade-off và tổng quát sang cấu hình chưa dùng khi tuning. Chỉ thêm penalty quen thuộc rồi tăng metric nhỏ là novelty yếu.

### B — Memory hữu ích đến từ quantum dynamics hay classical feedback? (ưu tiên 2 cho temporal)

**Câu hỏi:** Trong photonic temporal pipeline có cùng history window, feedback và readout, quantum feature map thêm thông tin dự đoán gì vượt classical features?

**Prior art:** J3, J7, J8 và P5 đã xử lý nhiều phần. Không đề xuất “thêm feedback” như đóng góp mới.

**Thiết kế ứng viên:** so sánh cùng memory module với features từ single photon, distinguishable/indistinguishable photons, coherent-intensity model, polynomial/random features. Giữ history và tuning budget đồng nhất; tách one-step prediction khỏi autonomous rollout. Xác định loại tương tác thời gian và miền tài nguyên mà khác biệt có hoặc không có.

**Giới hạn:** một cloud sampler không tự cung cấp optical delay, coherent memory hay fast feedforward. Dùng classical leaky memory thì gọi đúng là hybrid; không tuyên bố quantum memory. Chưa có QPU chỉ xác nhận được kết quả mô phỏng, không phải streaming latency thực tế.

### C — Chuyển mô hình giữa các đợt hiệu chuẩn, có tính giá recalibration (ưu tiên sau A)

**Câu hỏi:** Reservoir đã huấn luyện hôm trước còn dùng được ở lần chạy sau không, và cần bao nhiêu calibration samples/shots để khôi phục chất lượng?

**Prior art:** P4 đã có perturbation-aware training; P1 đã bàn calibration drift. Vì vậy noise augmentation không mới.

**Phạm vi ứng viên:** phân biệt sai số độc lập theo sample với drift có tương quan thời gian, phase offsets, loss không đều và distinguishability thay đổi. So sánh không thích nghi, domain randomization, refit readout, anchor-based alignment. Chi phí cập nhật tính vào tổng ngân sách.

**Điểm cần kiểm chứng:** chỉ là gap ứng viên sau tìm kiếm có giới hạn; chưa có bằng chứng phủ định mọi công trình trước. Mô phỏng không xác nhận distribution drift của máy thật. Trước hardware claim phải có calibration logs và data nhiều session.

### D — Molecular/material reservoir dưới domain shift và label budget (ứng dụng phụ)

**Câu hỏi:** Lợi ích ít mẫu có tồn tại khi test trên họ cấu trúc chưa thấy, hay chỉ trong random splits dễ hơn?

**Prior art:** J5 đã làm molecular QRC ít mẫu; P7 đã nghiên cứu small-data hardware và regularization. Đổi neutral atoms sang photonics, tự thân, chưa đủ novelty.

**Protocol ứng viên:** chọn dataset có cấu trúc/metadata cho split theo scaffold hoặc họ vật liệu; mọi preprocessing/feature selection fit trong training fold. So sánh mô hình cùng descriptor, classical random features, kernel/RF/boosting và baseline chuyên ngành. Báo sai số, calibration của uncertainty và budget; không cam kết coverage OOD nếu không có giả định phù hợp.

**Giới hạn:** static molecular descriptors không phải temporal QRC. Muốn temporal chemistry phải có dữ liệu động học/quá trình/molecular dynamics thật và câu hỏi tương ứng. Chưa chọn dataset, chưa xác minh novelty riêng ở từng property.

## 5. Ưu tiên cho Quanova

| Hướng | Bắt đầu không QPU | Gần Quandela | Rủi ro chính |
|---|---|---|---|
| A: finite-budget readout/correlation | Tốt, bằng mô phỏng và dữ liệu tác giả | Cao nếu Fock/PNR | Dễ trùng denoising/feature selection có sẵn |
| B: tách quantum features và memory | Tốt cho ablation nhỏ | Cao nhưng phụ thuộc feedback API | Thắng do classical memory, không do quantum |
| C: cross-session drift | Mô phỏng được; validation cần máy/logs | Cao | Noise giả định không giống máy thật |
| D: molecular/material shift | Tốt nếu có dữ liệu phù hợp | Trung bình; phải thiết kế encoding | Domain expertise và baseline mạnh |

Đề xuất không làm đồng thời cả bốn. Chọn A làm câu hỏi phương pháp; dùng B nếu chốt chuỗi thời gian, hoặc D nếu chốt vật liệu/phân tử. C là bước mở rộng khi có logs hoặc QPU.

Tiêu đề làm việc, chưa phải paper đã có novelty: **Measurement-budget-aware photonic reservoir learning: useful correlations under realistic sampling**.

Nếu ưu tiên đối tác vật liệu, gắn với một property cụ thể sau khi audit dataset. Nếu ưu tiên temporal industry, chọn một dữ liệu quá trình hoặc sensing có chia tập theo run/machine/time; tránh coi Lorenz/NARMA là bằng chứng công nghiệp.

## 6. Những gì có thể làm ngay mà không có QPU

1. Chốt temporal QRC hay stateless QELM trước khi chọn dataset.
2. Tái lập mô hình nhỏ, kiểm tra ideal và sampled features hội tụ; không triển khai simulator khổng lồ để khoe scale.
3. Thiết lập baseline cổ điển mạnh trước, chia dữ liệu đúng miền sử dụng.
4. So sánh exact probabilities với multinomial counts, photon loss và partial distinguishability ở cấu hình simulator hỗ trợ; thêm phase drift với giả định công khai khi cần.
5. Chỉ mở rộng khi một hypothesis sống sót qua ablation; nếu classical surrogate giải thích hết thì không gọi quantum advantage.

Tài nguyên đã tìm thấy:

- [MerLin ReservoirClassifier 0.4](https://merlinquantum.ai/0.4/user_guide/models/reservoir_classifier.html): frozen photonic feature map và readout cổ điển. Là điểm xuất phát static, không chứng nhận native temporal memory. Chưa cài/chạy trong lượt này.
- [QuEra molecular QRC code](https://github.com/QuEraComputing/robust-qrc-molecular-prediction): hữu ích tái lập pipeline và so sánh embedding; không phải mạch chạy nguyên trạng trên Quandela.
- [GBS experimental data](https://zenodo.org/records/15288036): có processed data/scripts và raw frames; raw archive lớn khoảng 41.8 GB, chưa tải. Có thể kiểm tra data format trước khi đề xuất subsampling; dataset này không thay thế thông số máy Quandela.

## 7. Tiêu chí dừng/đổi hướng

- Lợi ích mất khi baseline được tuning công bằng hoặc dùng cùng history: ghi nhận, không cherry-pick split.
- Chỉ có lợi ích ở exact expectation và mất dưới mọi budget khả thi: cần sửa mục tiêu hoặc xác định ngưỡng tài nguyên, không gọi near-term utility.
- Tính shots tiết kiệm nhưng bỏ chi phí discarded events/postselection, calibration hoặc model search: kết luận resource advantage chưa hợp lệ.
- Đóng góp chỉ là ghép một phương pháp preprocessing có sẵn với reservoir có sẵn: cần cơ chế/bound/benchmark tổng quát hơn; không mặc định đủ Q1.
- Mô phỏng quy mô nhỏ có thể đánh giá giả thuyết nhưng không chứng minh computational speedup. Nếu tận dụng được classical simulator/surrogate tốt, phải đưa nó vào đối chứng.

## 8. Giới hạn của bản khảo sát

Theo khung literature-review, bản này ưu tiên thematic synthesis và truy vết nguồn; không dựng PRISMA hoặc số lượng screening giả. Truy xuất publisher, arXiv, PubMed/PMC, kiểm tra journal DOI bằng Crossref. Không dùng bài đăng mạng xã hội làm bằng chứng đồng thuận quốc tế; các hướng cộng đồng ở đây là hướng thể hiện qua công trình, không phải khảo sát ý kiến đại diện.

Không đọc toàn bộ supplement/proof của mọi bài, chưa chạy code, chưa có kết quả mới. Các gap là câu hỏi ứng viên được đặt cạnh prior art gần nhất, chưa bảo đảm tính mới. Cần literature matrix sâu hơn trước khi chốt proposal.

Scientific-schematics thiếu OPENROUTER_API_KEY nên dùng bảng/sơ đồ chữ. Tài liệu là working research note Markdown, không xuất PDF trong lượt này. DOI verification report được lưu cạnh manifest; xác minh metadata không xác minh tính đúng của các kết luận.
