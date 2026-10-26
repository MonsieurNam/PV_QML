# Đánh giá năm hướng QRC cho Quanova

Ngày: 26/09/2026. Targeted narrative assessment; không phải systematic review hay chứng nhận novelty. Đã đối chiếu publisher, arXiv, OpenAlex/PubMed và kiểm tra DOI bằng Crossref. Chưa chạy mô phỏng, tái lập code hoặc kiểm tra đầy đủ chứng minh. Những cơ hội dưới đây là giả thuyết cần kiểm tra, không phải khoảng trống đã được xác nhận.

## Kết luận quyết định

Cả năm hướng có ý nghĩa khoa học, nhưng mô tả ban đầu đánh giá quá cao tính mới. Noise engineering, weak measurements và reservoir-based shadows đều có tiền lệ trực tiếp. Quantum-input temporal learning có từ năm 2021. Topology cũng đã được khảo sát có hệ thống. Không nên bắt đầu từ giả định một lĩnh vực còn trống.

Với một người chưa có QPU, thứ tự triển khai đề xuất: (1) damping với chi phí đo thực tế; (2) topology có chuẩn hóa tài nguyên; (5) readout theo tác vụ nếu đủ nền thống kê lượng tử; sau đó mới (3) monitoring/weak measurements hoặc (4) quantum-native sensing với cộng tác vật lý.

Thứ tự này tối ưu rủi ro dự án đầu tiên, không phải bảng xếp hạng tầm quan trọng khoa học. QRC temporal khác quantum extreme learning trên mẫu tĩnh: phải chỉ ra bộ nhớ nằm trong trạng thái lượng tử, feedback cổ điển hay cửa sổ đầu vào.

## Bảng so sánh và thời gian

Giả định: một người Python/ML khá, lượng tử cơ bản, dành khoảng 20 giờ/tuần, có cố vấn chuyên môn hàng tuần; bắt đầu mô phỏng hệ nhỏ. Các mốc là ước tính quản trị của người tổng hợp. Bản thảo chỉ khả thi nếu tìm được đóng góp; không bao gồm chờ phản biện, xin QPU hoặc chế tạo lab. Với 8–10 giờ/tuần, dự trù khoảng 1.5–2 lần thời gian, nhưng học nền và phụ thuộc cộng tác có thể lâu hơn.

| Hướng | Tiềm năng sau thu hẹp | Khó /5 | Pilot tối thiểu | Bản thảo có đóng góp, có điều kiện |
|---|---|---:|---|---|
| 1. Nhiễu/damping | Tốt cho đề tài đầu; ý tưởng chung đã đông | 3–4 | 4–6 tuần | 6–9 tháng |
| 2. Topology | Khá; cần nguyên lý dự báo hơn bảng so sánh | 3–4; Lie sâu: 5 | 4–8 tuần | 6–10 tháng; định lý Lie: 9–15+ |
| 3. Weak measurements | Cao về vật lý; cạnh tranh trực tiếp mạnh | 5 | 6–10 tuần | 9–15+ tháng |
| 4. Quantum-input/sensing | Cao dài hạn; phụ thuộc truy cập dữ liệu/hệ | 4–5 | 6–10 tuần, hệ đơn giản | 8–15+ tháng |
| 5. Readout/shadows | Cao nếu giảm được chi phí end-to-end | 4–5 | 5–8 tuần | 6–12 tháng |

Pilot là tái lập hoặc thử nghiệm nhỏ kiểm tra giả thuyết, không phải kết quả mới hay bài đã sẵn sàng nộp. Dự đoán phân tử phức tạp và xây nguồn quantum input vật lý có thể vượt xa các mốc này.

## 1. Noise engineering và fading memory

### Hiện trạng

[Taking advantage of noise in quantum reservoir computing, Scientific Reports 2023](https://doi.org/10.1038/s41598-023-35461-5) đã so sánh kênh nhiễu và ghi nhận ích lợi của amplitude damping trong thiết lập cụ thể. [Controllable damping, npj QI 2026](https://doi.org/10.1038/s41534-026-01229-8) đã có damping điều chỉnh được và thí nghiệm IBM. [Non-Markovianity and memory enhancement, npj QI 2026](https://doi.org/10.1038/s41534-026-01257-4) tiếp tục nghiên cứu cơ chế kéo dài bộ nhớ. Vì vậy không thể dùng tính mới “noise có ích” làm đóng góp chính.

### Sửa khái niệm

Fading memory không có nghĩa tăng tốc độ quên vô hạn. Cần quên điều kiện ban đầu nhưng giữ đủ lịch sử liên quan tác vụ. Damping quá mạnh có thể xóa tín hiệu; cực trị tối ưu phụ thuộc input timescale, encoding, topology, readout, task và ngân sách shots. “Tối ưu” phải chỉ rõ là optimum thực nghiệm trong miền khảo sát hay optimum được chứng minh.

Một điều kiện đủ kiểu contraction: nếu cùng chuỗi input cho hai trạng thái ban đầu dẫn tới khoảng cách giảm như q^t, 0<q<1, thì q đặt thang thời gian quên. Đây không phải điều kiện cần cho mọi kiến trúc, cũng không tự bảo đảm accuracy cao. Nhiễu quá trình, lỗi đọc và sai số finite shots phải được tách riêng.

### Cơ hội có điều kiện

Kiểm tra vùng lợi ích của damping có còn khi giữ cùng ngân sách shots và tính chi phí measurement/reset; đánh giá độ bền của cấu hình chọn trên validation khi thời gian tương quan tín hiệu hoặc noise drift thay đổi. Muốn vượt benchmark thuần túy cần cơ chế, bound hoặc quy tắc thiết kế tổng quát đã được kiểm chứng ngoài tập dùng để chọn tham số.

IBM và Quandela không tương đương: qubit relaxation/dephasing khác loss, nguồn đa photon, độ không phân biệt và sai số pha trong photonics. [Perceval NoiseModel](https://perceval.quandela.net/docs/v0.13/reference/noise_model.html) liệt kê các thông số nguồn/đường quang. Mất photon khỏi dual-rail code space không thể tự động thay bằng qubit amplitude damping trong code space. Với postselection phải báo tỷ lệ thành công và số lần phát thực tế.

## 2. Topology, graph và Lie algebra

### Hiện trạng

[Quantum reservoir computing on random regular graphs, PRA 2025](https://doi.org/10.1103/gq9r-d5q8) đã xét liên hệ connectivity, disorder, interactions, memory và hiệu năng. Chưa tìm được một triển khai Kagome đúng từng điều kiện đề xuất không có nghĩa chưa ai làm. Một tên topology khác không đủ chứng minh novelty.

### Sửa khái niệm

Tiến hóa unitary và kênh CPTP cố định tuyến tính theo density matrix. Tính phi tuyến theo dữ liệu cổ điển có thể xuất hiện từ cách input điều khiển kênh/encoding, lịch sử tương tác và readout/feedback; không phải graph tự tạo phép tiến hóa phi tuyến của quantum state.

Lie algebra của bộ generator cho thông tin về khả năng điều khiển/đối xứng. Reservoir cố định không nhất thiết khai thác toàn bộ algebra; dimension lớn không bảo đảm memory tốt hay task error thấp. Cần chỉ rõ accessible input controls, Hamiltonian và observable algebra.

### Cơ hội có điều kiện

Xây quan hệ dự báo giữa graph descriptors/symmetry và memory–nonlinear processing khi giữ cùng tài nguyên. Đối chứng hai chế độ: cùng tổng interaction strength cho mô hình vật lý, và cùng gate/depth budget sau mapping cho mạch thực. Thêm rewired graph cùng degree để tách ảnh hưởng connectivity khỏi số cạnh. Trên IBM cần tính SWAP/idle time; trên photonics cần đếm mode, interferometer depth và loss, không chỉ số CNOT.

Rủi ro lớn: kết quả chỉ phản ánh số cổng hoặc số tham số nhiều hơn; finite-size Kagome quá nhỏ không biểu hiện cơ chế vật lý kỳ vọng. Đi từ graph benchmark trước, chỉ dùng Lie nếu nó giải thích/dự báo được một quan sát cụ thể.

## 3. Weak measurements và partial trace

### Hiện trạng

[Mujal et al., npj QI 2023](https://doi.org/10.1038/s41534-023-00682-z) đã so sánh protocol weak/projective và trade-off readout–memory. [Feedback-driven QRC, PRX Quantum 2024](https://doi.org/10.1103/PRXQuantum.5.040325) là đối chứng quan trọng. [Harnessing Quantum Backaction, PRX 2026](https://doi.org/10.1103/j7f9-hfsj) đã tối ưu measurement strength cho temporal processing. Hướng này có giá trị nhưng không mới ở mức ý tưởng tổng quát.

### Sửa khái niệm

Partial trace rho_R=Tr_A(rho_RA) là mô tả bỏ subsystem A, không phải một kết quả đo. Nếu đo A bằng Kraus K_m thì xác suất p_m và trạng thái R có điều kiện phụ thuộc m. Không giữ m sẽ cho trạng thái không điều kiện; muốn đưa thông tin vào readout phải mô hình cả measurement instrument và dữ liệu m.

Weak measurements giảm nhiễu loạn mỗi phép đo nhưng thường thu ít thông tin hơn. Không thể đọc miễn phí một trạng thái lượng tử tùy ý mà giữ nguyên nó. Quét nhiều mức measurement strength không tự là định lý.

### Cơ hội có điều kiện

Trade-off thông tin–bộ nhớ có tính detector inefficiency, ancilla/reset cost, finite trajectories và feedback latency; adaptive monitoring phải tính cả chi phí training controller. Các đối chứng phải dùng cùng số bản sao/lần chạy và không dùng expectation values chính xác như dữ liệu miễn phí.

[IBM dynamic circuits](https://quantum.cloud.ibm.com/docs/en/tutorials/edc-cut-bell-pair-benchmarking) cung cấp measurement/feedforward, nhưng weak measurement thực hiện qua ancilla cần kiểm tra hỗ trợ và overhead của backend cụ thể. Không suy ra quyền truy cập Quandela Cloud bao gồm QND readout, optical loops hoặc quantum state persistence giữa jobs. Muốn thí nghiệm có thể cần cộng tác lab thay vì API thông thường.

## 4. Quantum-input temporal learning và sensing

### Hiện trạng

[Learning Temporal Quantum Tomography, PRL 2021](https://doi.org/10.1103/PhysRevLett.127.260401) đã học xử lý theo thời gian của quantum state streams. Preprint [hybrid temporal quantum-state processing, 19/06/2026](https://arxiv.org/abs/2606.21327) và [in-sensor sensing sự kiện không lặp lại, 05/09/2026](https://arxiv.org/abs/2609.06292) là đối thủ nghiên cứu mới, chưa coi là bằng chứng đã phản biện.

### Sửa khái niệm

Quantum-native input chỉ giảm hoặc bỏ classical encoding khi trạng thái vật lý được đưa trực tiếp vào reservoir. Chuỗi kỳ vọng năng lượng sinh bởi simulator rồi nhập vào QRC vẫn là classical input. Với dữ liệu quantum thật, state preparation, coupling/transduction, số bản sao, backaction và output estimation vẫn là nút thắt. Dự đoán đầy đủ state của phân tử phức tạp không được miễn chi phí output hàm mũ.

### Cơ hội có điều kiện

Bắt đầu từ dự báo vài observable của hệ spin nhỏ, ước lượng tham số drift hoặc denoising tín hiệu sensor dưới nguồn lực cố định. So với mô hình vật lý đã biết và Bayesian/Kalman filter khi giả định phù hợp, bên cạnh ESN/ML. Hệ sinh dữ liệu cần có thông số/noise ngoài phân phối huấn luyện để tránh inverse crime. Nếu muốn lợi thế quantum-native, phải có quantum interface thật hoặc mô hình interface có thể hiện thực hóa.

Ý nghĩa công nghiệp dài hạn có thể lớn cho sensor/device monitoring; ngắn hạn chưa chắc phù hợp Quanova nếu chỉ có quyền cloud QPU. Không đồng nhất tất cả bài có từ “quantum reservoir sensing” với QRC machine learning.

## 5. Higher-order correlations và shadow readout

### Hiện trạng

[Estimating many properties… via quantum reservoir processing, PRR 2024](https://doi.org/10.1103/PhysRevResearch.6.013211) đã nối reservoir với ước lượng nhiều thuộc tính. [Robust shallow shadows, Nature Communications 2025](https://doi.org/10.1038/s41467-025-57349-w) nghiên cứu hiệu quả phép đo dưới noise. [Photonic entanglement witnessing, Science Advances 2025](https://doi.org/10.1126/sciadv.ady7987) có demo QELM photon; không phải bằng chứng bộ nhớ temporal. Vì vậy không gọi đây là vùng đất trống.

### Sửa khái niệm

Theo [classical shadows, Nature Physics 2020](https://doi.org/10.1038/s41567-020-0932-7), sample complexity phụ thuộc shadow norm, không chỉ log số observable. Random local-Pauli có hệ số 3^k cho Pauli weight k. Bound cần các copies độc lập và tập observable cố định. Đừng đồng nhất số measurement settings với số shots hay chi phí postprocessing.

### Cơ hội có điều kiện

Readout theo tác vụ: chọn observable và phân bổ shots để giảm prediction risk, thay vì khôi phục tất cả tương quan. Đối chứng local shadows, grouped commuting measurements, shallow/derandomized schemes và low-order readout. Feature selection chỉ trên train/validation; tránh dùng kết quả trên test để chọn observables. Lợi ích của sampling phải vượt chi phí hiệu chuẩn và postprocessing.

Với reservoir online, không tự coi các timestep hoặc các trajectory có điều kiện khác nhau là copies của cùng rho_t. Tái chuẩn bị rho_t có thể cần replay lịch sử input; đó là chi phí thực và có thể không khả thi cho sự kiện không lặp lại.

## Protocol pilot đề xuất: một trục chính, không ôm cả năm hướng

Mục tiêu đầu: kiểm tra lợi ích damping ở mức prediction risk/measurement budget, không tuyên bố optimum phổ quát. Chọn một reservoir và một readout cố định, dùng topology như control. Chỉ thêm shadows khi đo được readout là nút thắt thật.

- Tuần 1–2: đọc/tái lập paper noise và finite-shot gần nhất; viết novelty matrix và ngân sách thực. Xác nhận backend là tùy chọn kiểm chứng, không mặc định có sẵn.
- Tuần 3–4: dựng mô phỏng nhỏ 4–6 qubit, baseline ESN/ridge với lag features; finite-shot sampling hợp lệ. Không có thí nghiệm nào đã được chạy trong bản đánh giá này.
- Tuần 5–6: so sánh zero/weak/strong damping, readout noise tách riêng; vài task có timescale khác nhau; nhiều reservoir/input/measurement seeds.
- Tuần 7–8: hold-out task/noise drift, uncertainty ở cấp chuỗi độc lập; chốt go/no-go và không tinh chỉnh trên test.

Tiếp tục nếu hiệu ứng bền qua seed, budget và hold-out, có cơ chế giải thích và khác công trình gần nhất. Thu hẹp hoặc đổi hướng nếu lợi ích chỉ có ở exact expectation, một seed, leakage, nhiều shots hơn, hoặc là reproduction của kết quả đã biết. Kết quả âm chỉ đáng công bố nếu cung cấp giới hạn/quy luật có tính khái quát.

## Giới hạn và tài liệu

Không có cơ sở để gán phần trăm accepted hoặc hứa PRL/PRX từ tên topic. Timeline không phải xác suất thành công. Đã tìm thấy prior art trực tiếp đủ bác bỏ nhận định “chưa ai làm”, nhưng chưa đủ xác nhận mọi gap đề xuất còn mới. Skill scientific-critical-thinking được dùng để kiểm tra giả định, tài nguyên và confounding; không dùng hệ phân hạng bằng chứng y khoa cho định lý vật lý.

Scientific-schematics không có cấu hình dịch vụ; bảng so sánh thay cho hình AI. Không tạo PDF vì yêu cầu là đánh giá nghiên cứu, không phải bản thảo xuất bản.

[Danh mục nguồn và chuỗi truy vấn](sources/QRC_five_directions_20260926_references.md).

Kiểm tra metadata: 14/14 DOI journal được xác minh qua DOI resolver/Crossref. Ba preprint được ghi riêng với ngày nộp, không tính thành bài journal.
