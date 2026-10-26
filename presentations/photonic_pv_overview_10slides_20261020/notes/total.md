# 1_P01_overview

Em xin báo cáo hướng photonic QML cho dự báo xác suất công suất điện mặt trời trước mười lăm phút. Câu hỏi chính là liệu đặc trưng tạo bởi hệ quang học lượng tử có bổ sung thông tin dự đoán hữu ích so với các cách xử lý cổ điển hay không. Luồng bên phải cho thấy vai trò dự kiến của thành phần quang học: nhận lịch sử PV, biến đổi thành đặc trưng, rồi đưa vào mô hình dự báo quantile.

Tại ngày hai mươi tháng mười, dự án đã có dữ liệu thật và pipeline để thực hiện phép thử. Bằng chứng thực thi gần nhất được lưu ngày mười sáu tháng mười. Full benchmark trên dữ liệu thật vẫn chưa chạy; vì vậy báo cáo này trình bày cơ sở lựa chọn, thiết kế và tiến độ, chưa đưa ra kết luận quantum advantage hay tốc độ phần cứng. Thông điệp cần giữ là dữ liệu đã sẵn sàng, còn giá trị của mô hình phải được kiểm chứng.

---

# 2_P02_motivation

Động cơ ban đầu là tìm một hướng QML phù hợp với nhu cầu nghiên cứu công nghiệp của Quanova. Hướng đó phải đồng thời có ba yếu tố: một target đo được với dữ liệu có nguồn gốc rõ; một giả thuyết khoa học có thể kiểm tra và bác bỏ; và một thử nghiệm khả thi khi chưa có QPU. Vì vậy, dự án bắt đầu bằng mô phỏng nhỏ trước khi đề nghị đầu tư phần cứng.

Các ứng viên đã khảo sát gồm tài chính, tín hiệu và bất thường thiết bị, tính chất phân tử hoặc vật liệu, cùng các bài toán chuỗi thời gian. Ảnh bên phải là hệ Roof Tilted Array thật tại NIST, lấy từ trang chính thức nist.gov, giúp gắn bài toán dự báo với một hệ thống đo thực tế. Đây không phải hệ thống do Quanova sở hữu. Phần này dựa trên hồ sơ industry-first QML ngày hai mươi sáu tháng chín. Từ các điều kiện ban đầu này, nhóm tiếp tục xem literature để biết phải kiểm chứng điều gì.

---

# 3_P03_literature

Khảo sát literature dẫn đến ba thay đổi thiết kế. Thứ nhất, công trình Better than classical của Bowles và cộng sự nhấn mạnh việc dùng baseline cổ điển mạnh và đối chứng công bằng. Nếu chỉ so với một mô hình yếu thì kết quả tốt hơn chưa đủ nói lên vai trò của lượng tử.

Thứ hai, các nghiên cứu photonic kernel năm hai nghìn không trăm hai mươi lăm và forecasting nhiều photon của Di Bartolo năm hai nghìn không trăm hai mươi sáu cho thấy cần tách đóng góp của giao thoa khỏi nguồn bộ nhớ. Bởi vậy thiết kế hiện tại phải có đối chứng photon phân biệt được và ghi rõ bộ nhớ nằm trong cửa sổ cổ điển.

Thứ ba, Ahmed đã nghiên cứu finite sampling, còn Mills và Mezher đã nghiên cứu giảm ảnh hưởng mất photon. Những hướng này có tiền lệ, nên không thể coi quét shots hoặc thêm loss là novelty tự thân. Sagingalieva và cộng sự cũng đã công bố QML cho dự báo PV. Damping, topology và weak measurement đều đã có nghiên cứu trước. Các khảo sát ở đây có mục tiêu, không phải systematic review toàn ngành; danh mục đầy đủ được lưu trong sources/Photonic_PV_pilot_20260926_references.md của repository.

---

# 4_P04_hypothesis

PV được chọn làm phép thử đầu tiên vì có công suất và thời tiết đo thật công khai, target ngắn hạn có thể định nghĩa chính xác, và nhiều baseline xác suất rõ ràng. Những điều này giúp thực hiện một phép so sánh tái lập được, thay vì chọn ứng dụng chỉ vì có tên lượng tử.

Giả thuyết là đặc trưng của photon không phân biệt được có thể cải thiện chất lượng dự báo vượt các feature maps cổ điển và phiên bản photon phân biệt được. Giả thuyết này hoàn toàn có thể không được dữ liệu ủng hộ. Phạm vi đang triển khai chỉ là mô phỏng exact với hai photon và bốn mode. Finite shots, loss, chuyển site và phần cứng là các bước sau, có điều kiện. Chưa có cơ sở để nói tổ hợp đề xuất đã được xác nhận mới hoặc chắc chắn tạo lợi ích. Căn cứ thiết kế nằm trong Photonic_PV_pilot_review_20260926.md và cấu hình NIST đã được ghi phê duyệt.

---

# 5_P05_architecture

Sơ đồ đi từ input ở trên, qua các nhánh mô hình, đến output chung bên phải. Input matched gồm công suất hiện tại, công suất trước mười lăm phút, bức xạ và nhiệt độ. GHI là tên trường pipeline; cảm biến NIST thực tế là silicon reference cell nằm ngang. Scaling chỉ fit trên train. Bộ nhớ nằm trong lag cổ điển, không có trạng thái lượng tử lưu giữa các timestamp.

Nhánh đề xuất dùng hai photon trên bốn mode: Fock input qua V một cố định, pha phụ thuộc dữ liệu, rồi V hai cố định. Mười xác suất đo exact từ Perceval SLOS trở thành features cho quantile head huấn luyện cổ điển. Năm seeds kiểm tra độ nhạy mạch. Nhánh control đổi sang photon phân biệt được nhưng giữ circuit, input và chính sách readout. Nhánh cổ điển gồm raw inputs hoặc mười random Fourier features qua quantile head, cùng quantile XGBoost và persistence. XGBoost dùng sáu lag từ không đến sáu mươi phút, weather hiện tại và hình học mặt trời: đây là baseline ứng dụng mạnh, không phải đối chứng cùng input. Persistence dùng công suất hiện tại cộng phân vị residual fit trên train.

Output gồm bảy quantile công suất: 0.05, 0.10, 0.25, 0.50, 0.75, 0.90 và 0.95. Target là trung bình bin năm phút kết thúc ở t cộng mười lăm, không phải trung bình cả mười lăm phút tiếp theo. q50 là trung vị; q05 đến q95 tạo khoảng chín mươi phần trăm. WIS và skill đánh giá chất lượng; coverage và width đánh giá calibration và độ rộng; runtime và RAM đánh giá chi phí. Đây là kiến trúc và phép thử dự kiến, chưa có điểm số mô hình thật.

---

# 6_P06_evaluation

Thiết kế có hai phép so sánh vì chúng trả lời hai câu hỏi khác nhau. Giá trị ứng dụng được kiểm tra bằng persistence, raw inputs với quantile head, random Fourier features với cùng head, và quantile XGBoost dùng các lag mạnh hơn. Đóng góp của giao thoa được kiểm tra bằng photon không phân biệt và photon phân biệt trên cùng circuit, đầu vào và chính sách readout.

Train và validation phục vụ scaling, lựa chọn siêu tham số và chọn baseline B. Chỉ sau khi khóa lựa chọn mới đánh giá holdout năm hai nghìn không trăm mười bảy. WIS, tức weighted interval score, là chỉ số chính, chuẩn hóa theo công suất định mức và thấp hơn tốt hơn. WIS kết hợp độ rộng khoảng với mức phạt khi target nằm ngoài khoảng. Skill bằng một trừ WIS của system chia WIS của B; dương nghĩa là tốt hơn B. System bao gồm chính sách fallback đã đăng ký khi nhánh photonic thất bại, không chỉ các timestamp thành công. Coverage chín mươi phần trăm đo tỷ lệ quan trắc nằm trong q05 đến q95; width là độ rộng khoảng chia công suất định mức. Chi phí ghi wall-clock runtime và peak RAM. Paired bootstrap dùng khối một ngày, hai nghìn lần lặp, kiểm tra nhạy với khối hai và bảy ngày để đánh giá bất định có phụ thuộc thời gian.

Theo protocol, điều kiện đi tiếp gồm cận dưới khoảng tin cậy chín mươi lăm phần trăm của relative system WIS skill lớn hơn hai phần trăm, coverage của khoảng chín mươi phần trăm nằm trong tám mươi lăm đến chín mươi lăm phần trăm, và tài nguyên đạt gate. Đây là tiêu chí đã đăng ký, chưa phải kết quả mô hình.

---

# 7_P07_data

Dữ liệu chính là NIST Roof, gồm AC power, bức xạ và nhiệt độ môi trường. Metadata AC là bảy mươi lăm kilowatt; không nhầm với công suất DC của các panel. Nguồn một phút được gộp thành bin năm phút chỉ khi đủ quan sát, không nội suy. Sau điều kiện daylight, lịch sử, target và continuity, train năm hai nghìn không trăm mười lăm có năm mươi mốt nghìn bốn trăm linh năm mẫu; validation năm tiếp theo có bốn mươi bảy nghìn bốn trăm tám mươi bảy; test năm hai nghìn không trăm mười bảy có bốn mươi ba nghìn tám trăm linh sáu.

Đồ thị là hai mươi lăm quan trắc liên tục của ngày mùng một tháng sáu năm hai nghìn không trăm mười lăm, chỉ lấy từ train. Nét liền là lịch sử sẵn có trước hoặc tại mười hai giờ; nét đứt là quan trắc về sau được hiện lại để giải thích nhãn, không phải prediction và không được dùng làm input. Khi issue lúc mười hai giờ, target là công suất trung bình của bin từ mười hai giờ mười đến mười hai giờ mười lăm: năm mươi mốt phẩy bảy mốt kilowatt trong ví dụ này. Giờ nguồn là EST cố định rồi chuyển UTC. Delay bằng không chỉ là giả định offline. Nguồn và hash được lưu trong train_example_provenance.json; kênh bức xạ là silicon reference cell, không phải thermopile pyranometer.

---

# 8_P08_audit

Đây là bằng chứng đã đo về độ sẵn có dữ liệu. Hai dãy cột phân biệt coverage của dữ liệu canonical và coverage sau khi yêu cầu đủ mẫu xây dựng features. Train có các tỷ lệ xấp xỉ chín mươi tám phẩy chín và chín mươi sáu phẩy sáu phần trăm; validation là chín mươi bảy phẩy ba và tám mươi chín phần trăm; test là chín mươi bảy phẩy sáu và tám mươi hai phẩy ba phần trăm. Các split đều qua ngưỡng canonical chín mươi phần trăm và eligible tám mươi phần trăm.

Test còn có một trăm năm mươi bảy ngày đạt chuẩn và ba mươi lăm tuần đủ chuẩn không chồng lấn, vượt yêu cầu một trăm năm mươi ngày và hai mươi tuần. Một ngày usable phải đạt coverage eligible ít nhất chín mươi phần trăm; tuần dùng pooled coverage tám mươi phần trăm. Đây là data availability, khác coverage của prediction intervals ở slide trước. Audit authenticated ngày mười sáu tháng mười ghi PASS. Do complete issue log không có, dự án đã dùng amendment cho phép instrument history và QC định lượng thay thế; vẫn có thể còn lỗi chưa gắn nhãn. Audit không chứng minh hiệu quả forecasting hoặc khả năng suy rộng sang PV Việt Nam.

---

# 9_P09_status

Về tiến độ, nhóm đã xây dựng pipeline và kiểm tra end-to-end trên dữ liệu tổng hợp, chuẩn bị dữ liệu NIST thật, audit khả dụng, dựng chronological splits và profiling trên train. Checkpoint phần mềm đầu tháng mười ghi nhận bảy trăm hai mươi mốt tests passed và hai lượt synthetic end-to-end accepted, nhưng đó là bằng chứng phần mềm lịch sử, chưa phải scientific acceptance cho mô hình thật.

Controller của lần chạy ngày mười sáu tháng mười ghi chưa bắt đầu full training và chưa mở hiệu năng holdout. Kế hoạch thực thi ngoại suy benchmark toàn stage khoảng một trăm mười một giờ với đỉnh RAM khoảng ba mươi ba phẩy bốn GiB; baseline stage khoảng bảy mươi hai giờ. Các số này có hệ số dự phòng và có thể đếm chồng một số allocation. Chúng là ước lượng từ profiling, không phải thời gian full run đã đo và cũng không phải latency mỗi dự báo. Route hiện yêu cầu xác minh cloud capacity. Đây là nút thắt kỹ thuật, chưa phải kết luận mô hình tốt hoặc kém.

---

# 10_P10_next

Bước tiếp theo gồm xác minh capacity, chạy các mô hình và đối chứng đã khóa, rồi đánh giá năm hai nghìn không trăm mười bảy sau khi model selection hoàn tất. Sau đó mới ra quyết định GO, EXTEND hoặc STOP dựa đồng thời trên WIS, calibration và tài nguyên.

Nếu bằng chứng đủ rõ và phù hợp ngân sách, giai đoạn sau mới kiểm tra finite shots và photon loss, rồi cân nhắc xác minh phần cứng. Nếu chỉ tốt hơn photon phân biệt nhưng không cạnh tranh với baseline mạnh thì chưa đủ lập luận cho ứng dụng. Nếu đặc trưng cổ điển giải thích được lợi ích với chi phí thấp hơn thì cần ghi nhận điều đó. Kết luận mong muốn của pilot là một câu trả lời đáng tin, dù dương, âm hay chưa phân định được. Thành phần lượng tử cần chứng minh rằng nó xứng đáng có mặt trong pipeline.
