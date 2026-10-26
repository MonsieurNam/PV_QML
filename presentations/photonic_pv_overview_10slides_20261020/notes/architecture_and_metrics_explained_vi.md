# Giải thích kiến trúc và metric bằng ví dụ đơn giản

Ghi chú được lưu ngày 26/10/2026 từ phần giải thích sau báo cáo. Các số dự báo ở
ví dụ tính điểm là số giả định để học cách tính, không phải kết quả benchmark.

## Fock và đường đi của dữ liệu

Fock mô tả số photon ban đầu trên từng đường ánh sáng. Trong cấu hình hai photon,
bốn mode, input là `|1,1,0,0⟩`: đường một và hai có một photon, hai đường còn lại
không có photon. Trạng thái này giữ nguyên giữa các mẫu.

Dữ liệu sau scaling đi vào D để điều khiển pha. Cách đọc luồng đúng là:

```text
Dữ liệu → scaling về [0,1] → bốn góc pha
                                  ↓
Fock cố định → V1 → D dùng bốn góc pha → V2 → 10 xác suất
```

Vì vậy, không đọc mũi tên tổng quát trên slide thành phép chuyển trực tiếp dữ liệu
sau scaling sang số photon ở Fock input.

## V1, D và V2

V1 và V2 là hai bộ trộn ánh sáng, được code biểu diễn bằng ma trận unitary 4×4.
Mỗi ma trận được tạo theo seed bằng một phép sinh Haar-random unitary và giữ cố
định trong quá trình học readout. Năm circuit seeds kiểm tra độ nhạy theo mạch.
Nếu làm phần cứng, phép trộn có thể được triển khai bằng mạng beam splitter và bộ
chỉnh pha; mô phỏng hiện tại chưa phải bản thiết kế chi tiết mạng linh kiện.

D giống bốn núm chỉnh pha, mỗi đường một núm. Với vector đã scale x, code dùng
`phi_j = c*x_j` và `D = diag(exp(i*phi_j))`. Với x = [0,2; 0,4; 0,6; 0,8] và
c = pi, góc pha là [36; 72; 108; 144] độ. Cấu hình xem xét c = pi/2 hoặc pi.
D không trộn các đường; V2 chuyển ảnh hưởng của pha tương đối thành khác biệt
trong phân bố xác suất đầu ra. Theo quy ước vector cột, toàn mạch là `V2 @ D @ V1`.

Hai photon có 10 cấu hình đếm đầu ra trên bốn mode: 4 cấu hình cùng trên một
đường và 6 cấu hình trên hai đường khác nhau. Simulator tính xác suất của từng
cấu hình. Mười xác suất quang học là features, không phải quantile công suất PV.

## Quantile head

Nhánh photonic chuẩn hóa 10 đặc trưng xác suất bằng StandardScaler fit trên
train, rồi dùng bảy mô hình hồi quy tuyến tính phân vị độc lập. Không có lớp ẩn
hay MLP. Có thể biểu diễn gọn như một lớp affine 10 đầu vào, 7 đầu ra, với 70
trọng số và 7 bias; thực tế code fit riêng từng QuantileRegressor.

Mỗi đầu ra có dạng `q_tau = b_tau + sum_j(w_tau,j*z_j)`. Các mức tau là 0.05,
0.10, 0.25, 0.50, 0.75, 0.90 và 0.95. Mô hình tối ưu pinball loss với L1
regularization. Ví dụ q90: dự báo thấp hơn thực tế 5 kW bị phạt 0,9×5 = 4,5;
dự báo cao hơn 5 kW bị phạt 0,1×5 = 0,5. Các mức quantile khác nhau vì chúng
dùng mức phạt bất đối xứng khác nhau.

Pipeline giới hạn đầu ra trong [0, capacity] rồi sắp xếp mỗi hàng để bảo đảm
quantile không đảo thứ tự. q50 là trung vị dự báo; [q05,q95] là khoảng danh
nghĩa 90%, cần được kiểm chứng bằng coverage trên nhiều dự báo.

## Ví dụ chấm điểm

Giả sử dự báo [q05,q10,q25,q50,q75,q90,q95] = [40,42,46,50,54,58,60] kW,
quan trắc y = 52 kW, công suất định mức C = 75 kW.

Khoảng 90% là [40,60], bao phủ thành công y. Một lần thành công không chứng
minh coverage 90%. Nếu 90 trong 100 lần quan trắc nằm trong khoảng tương ứng,
coverage thực nghiệm là 90%. Width = 60−40 = 20 kW; normalized width = 20/75,
xấp xỉ 26,7% công suất định mức.

WIS kết hợp độ rộng và hình phạt khi quan trắc nằm ngoài khoảng. Với khoảng
[l,u] có mức danh nghĩa 1−a:

`IS_a = (u-l) + (2/a)*max(l-y,0) + (2/a)*max(y-u,0)`.

Với ví dụ y = 52, cả ba khoảng đều bao phủ y:

| Thành phần | Giá trị | Trọng số | Đóng góp |
| --- | ---: | ---: | ---: |
| Sai lệch trung vị | abs(52−50) = 2 | 0,5 | 1 |
| IS của khoảng 50%, [46,54] | 8 | 0,25 | 2 |
| IS của khoảng 80%, [42,58] | 16 | 0,10 | 1,6 |
| IS của khoảng 90%, [40,60] | 20 | 0,05 | 1 |

WIS = (1+2+1,6+1)/3,5 = 1,6 kW. Normalized WIS = 1,6/75 = 0,021333.
Đây là điểm phạt chuẩn hóa, không phải phát biểu sai số phần trăm thông thường.
Nếu y = 65, riêng IS của khoảng 90% tăng thành 20 + (2/0,1)×5 = 120.

Nếu WIS baseline là 2 và WIS mô hình là 1,6, skill = 1−1,6/2 = 20%:
giảm điểm phạt 20% so với baseline. Kết luận nghiên cứu phải dựa trên toàn bộ
tập đánh giá và khoảng tin cậy theo khối ngày, không dựa trên một ví dụ.
Runtime và peak RAM lần lượt đo thời gian chạy và mức bộ nhớ lớn nhất.

## Canonical và eligible

Canonical là dữ liệu đã đưa về khuôn chuẩn và qua kiểm tra chất lượng áp dụng.
Eligible là thời điểm có đủ dữ liệu để xây dựng một mẫu dự báo hoàn chỉnh.

Một dòng lúc 12:00 có thể hợp lệ, nhưng mẫu 12:00 vẫn không eligible nếu thiếu
lag 11:45 hoặc target 12:15. Pilot yêu cầu các interval liên tục từ t−60 đến
t+15, cùng điều kiện daylight, availability và ranh giới split. Dữ liệu tương
lai chỉ phục vụ tạo nhãn và kiểm tra mẫu; không trở thành input dự báo.

Nếu có 100 mốc ban ngày dự kiến, 95 mốc có canonical và 80 mốc có eligible,
hai coverage là 95/100 = 95% và 80/100 = 80%. Cả hai đo độ sẵn có dữ liệu,
khác coverage của khoảng dự báo và khác chất lượng dự báo.

## Đối chiếu triển khai và tài liệu

- [Fock input, V1/V2 và phase encoding](../../../src/pv_pilot/photonics.py).
- [Scaling input và điều kiện tạo mẫu](../../../src/pv_pilot/splits.py).
- [QuantileRegressor và WIS](../../../src/pv_pilot/baselines.py).
- [Scaling xác suất và readout photonic](../../../src/pv_pilot/benchmark.py).
- [Perceval: biểu diễn trạng thái](https://perceval.quandela.net/docs/v0.12/basics.html).
- [scikit-learn: QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html).
- [scoringutils: WIS](https://epiforecasts.io/scoringutils/reference/wis.html).
