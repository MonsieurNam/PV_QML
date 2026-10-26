# 1_P01_overview

Em xin trình bày tổng quan dự án ứng dụng photonic quantum machine learning, tức học máy lượng tử quang học, cho dự báo công suất điện mặt trời trước 15 phút.

Câu hỏi xuyên suốt là: liệu một phép biến đổi đặc trưng bằng hệ quang học lượng tử có giúp dự báo tốt hơn các mô hình cổ điển hay không, và nếu có thì lợi ích đó có tương xứng với chi phí thực hiện không?

Luồng trên slide thể hiện ý tưởng chung: từ lịch sử công suất và thông tin thời tiết, tạo ra các đặc trưng photonic, rồi dùng một mô hình được huấn luyện cổ điển để dự báo các phân vị công suất. Thay vì chỉ cho một giá trị, mô hình còn mô tả mức bất định của dự báo.

Báo cáo đi từ động cơ nghiên cứu, những bài học từ literature, lý do chọn bài toán PV, đến kiến trúc, dữ liệu và kế hoạch đánh giá. Theo mốc bằng chứng thực thi ngày 16 tháng 10 được dùng trong deck, dữ liệu và pipeline đã được chuẩn bị, nhưng full benchmark trên dữ liệu thật chưa chạy. Vì vậy, đây là báo cáo về cơ sở lựa chọn và mức độ sẵn sàng thực nghiệm; giá trị của thành phần lượng tử vẫn là câu hỏi cần kiểm chứng.

Trước hết, em xin bắt đầu từ lý do hình thành hướng nghiên cứu này.

---

# 2_P02_motivation

Động cơ ban đầu là tìm một hướng nghiên cứu QML có thể gắn với nhu cầu công nghiệp của Quanova. Từ đầu, nhóm đặt ra ba điều kiện lựa chọn.

Thứ nhất là tính hữu ích: phải có một đại lượng cần dự báo hoặc ra quyết định, đo được và có dữ liệu truy xuất được nguồn gốc. Thứ hai là giá trị khoa học: phải chỉ ra thành phần lượng tử được kỳ vọng đóng góp ở đâu và thiết kế được phép thử có thể bác bỏ kỳ vọng đó. Thứ ba là khả năng triển khai: thử nghiệm ban đầu phải làm được bằng mô phỏng nhỏ, khi chưa giả định có quyền truy cập bộ xử lý lượng tử.

Những nhóm ứng dụng đã được xem xét gồm tài chính, tín hiệu và bất thường thiết bị, tính chất phân tử hoặc vật liệu, cùng các bài toán chuỗi thời gian. Ở bước này, mục tiêu là tìm một bài toán đầu tiên đủ rõ để xây dựng phép kiểm chứng đáng tin.

Ảnh bên phải là hệ Roof Tilted Array thực tế tại NIST, nguồn cung cấp dữ liệu cho pilot. Ảnh giúp gắn nghiên cứu với một hệ thống đo vật lý cụ thể; đây không phải hệ thống do Quanova sở hữu.

Sau khi xác định các điều kiện lựa chọn, câu hỏi tiếp theo là các nghiên cứu trước đã gợi ý cách thiết kế thực nghiệm như thế nào.

---

# 3_P03_literature

Khảo sát tài liệu dẫn đến ba bài học chính và cả ba đều được đưa vào thiết kế hiện tại.

Bài học đầu tiên là phải benchmark công bằng. Công trình của Bowles và cộng sự về benchmarking QML nhấn mạnh vai trò của baseline cổ điển mạnh và đối chứng phù hợp. Một mô hình lượng tử tốt hơn một baseline yếu chưa đủ chứng minh đóng góp của thành phần lượng tử. Vì vậy, pilot cần cả đối chứng cùng đặc trưng và một baseline ứng dụng mạnh.

Bài học thứ hai là phải tách các cơ chế tạo ra kết quả. Các nghiên cứu photonic kernel và dự báo chuỗi thời gian bằng nhiều photon đặt ra những câu hỏi về giao thoa, biểu diễn dữ liệu và bộ nhớ. Trong thiết kế của dự án, thông tin lịch sử nằm trong các lag cổ điển. Đối chứng photon phân biệt được được bổ sung để kiểm tra riêng vai trò của giao thoa giữa các photon.

Bài học thứ ba là phải tính đến điều kiện thực hiện. Ahmed và cộng sự đã nghiên cứu finite sampling; Mills và Mezher nghiên cứu mất photon. Do đó, việc thêm shots hoặc loss tự nó chưa đủ tạo tính mới. QML cho dự báo PV cũng đã có công trình của Sagingalieva và cộng sự; damping, topology và weak measurement đều có tiền lệ.

Đây là khảo sát có mục tiêu, chưa phải systematic review toàn ngành. Kết quả quan trọng ở giai đoạn này là những yêu cầu thiết kế rút ra được; tính mới của tổ hợp đề xuất vẫn cần được đánh giá bằng bằng chứng.

---

# 4_P04_hypothesis

Từ các tiêu chí và bài học đó, PV được chọn làm bài toán thử nghiệm đầu tiên vì ba lý do.

Thứ nhất, có dữ liệu công suất và thời tiết đo thực tế được công bố, giúp truy xuất nguồn gốc và tái lập xử lý dữ liệu. Thứ hai, target ngắn hạn có thể định nghĩa rõ về thời điểm phát hành dự báo, thời gian dự báo trước và khoảng lấy trung bình. Thứ ba, bài toán có những baseline xác suất rõ ràng để so sánh.

Giả thuyết cụ thể là: đặc trưng tạo bởi các photon không phân biệt được có thể cải thiện chất lượng dự báo so với đặc trưng cổ điển và đối chứng photon phân biệt được. Đây là giả thuyết có thể không được dữ liệu ủng hộ. Nếu các đối chứng giải thích được toàn bộ lợi ích, thì cần ghi nhận kết quả đó.

Phạm vi hiện tại được giới hạn ở mô phỏng exact với hai photon và bốn mode. Exact nghĩa là lấy xác suất từ simulator, chưa chịu nhiễu do số lần đo hữu hạn. Finite shots, mất photon, thay đổi site và xác minh phần cứng là các bước sau, chỉ thực hiện khi kết quả bước đầu đủ thuyết phục.

Slide tiếp theo trình bày cách đưa giả thuyết này thành một kiến trúc có thể so sánh được.

---

# 5_P05_architecture

Em xin đi theo sơ đồ từ input ở phía trên, qua các nhánh mô hình, đến output bên phải.

Đầu vào matched gồm công suất tại thời điểm hiện tại, công suất trước 15 phút, bức xạ và nhiệt độ không khí. Các tham số chuẩn hóa chỉ được học trên tập train để tránh đưa thông tin tương lai vào quá trình xây dựng mô hình. GHI là tên trường trong pipeline; ở dữ liệu NIST này, kênh đo thực tế là silicon reference cell nằm ngang.

Nhánh đề xuất ở trên cùng bắt đầu bằng trạng thái Fock với hai photon trên bốn mode. Trạng thái đi qua lớp trộn cố định V một, lớp pha phụ thuộc dữ liệu D, rồi lớp trộn cố định V hai. Pha được đặt giữa hai lớp trộn để ảnh hưởng đến xác suất đầu ra. Simulator Perceval SLOS tính 10 xác suất tương ứng với các cấu hình đầu ra; vector này trở thành đặc trưng cho quantile head được huấn luyện cổ điển. Nhóm dùng năm circuit seeds để đánh giá độ nhạy theo mạch, không chọn seed tốt nhất sau khi xem test.

Nhánh giữa là đối chứng cơ chế: thay photon không phân biệt được bằng photon phân biệt được, nhưng giữ circuit, input và chính sách huấn luyện readout. Nhánh dưới gồm raw inputs hoặc 10 random Fourier features đi qua quantile head, cùng quantile XGBoost và persistence. Persistence lấy công suất hiện tại cộng các phân vị sai số được học trên train. XGBoost được cấp tập lag mở rộng từ 0 đến 60 phút, thời tiết hiện tại và hình học mặt trời; đây là baseline ứng dụng mạnh hơn về đầu vào.

Các nhánh cùng dự báo bảy phân vị công suất từ 0,05 đến 0,95. q50 là trung vị; q05 đến q95 tạo khoảng dự báo danh nghĩa 90%. Target là trung bình của bin năm phút kết thúc tại t cộng 15 phút. Bộ nhớ trong kiến trúc nằm ở các lag cổ điển; không có trạng thái lượng tử được giữ liên tục giữa các timestamp.

Với các nhánh này, bước tiếp theo là xác định chúng trả lời những câu hỏi gì và được chấm điểm ra sao.

---

# 6_P06_evaluation

Có hai phép so sánh cần tách rõ. Phép thứ nhất hỏi về giá trị ứng dụng: mô hình photonic có cạnh tranh được với persistence, raw inputs, random Fourier features và quantile XGBoost hay không? Phép thứ hai hỏi về cơ chế: trên cùng circuit, input và chính sách readout, photon không phân biệt được có tạo lợi ích so với photon phân biệt được hay không?

Chỉ số chính là WIS, tức weighted interval score, được chuẩn hóa theo công suất định mức. Chỉ số này kết hợp độ rộng khoảng dự báo với mức phạt khi quan trắc nằm ngoài khoảng; giá trị thấp hơn là tốt hơn. Baseline B được chọn bằng validation. Skill bằng một trừ WIS của system chia WIS của B, nên skill dương biểu thị cải thiện so với B. System bao gồm cả chính sách fallback đã đăng ký khi nhánh photonic thất bại.

Để kiểm tra độ tin cậy của khoảng dự báo, nhóm đo coverage thực nghiệm của khoảng q05 đến q95 và độ rộng khoảng đã chuẩn hóa. Khoảng quá rộng có thể dễ bao phủ dữ liệu nhưng ít hữu ích, nên phải xem hai đại lượng cùng nhau. Chi phí được theo dõi bằng thời gian chạy và peak RAM.

Các lựa chọn mô hình được khóa trước khi đánh giá test 2017. Khoảng tin cậy của skill được tính bằng paired bootstrap theo khối ngày để xét sự phụ thuộc thời gian. Điều kiện GO yêu cầu cận dưới khoảng tin cậy 95% của skill vượt 2%, coverage của khoảng 90% nằm trong 85–95%, và các gate tài nguyên đạt. Đây là tiêu chí đã đặt ra, chưa phải kết quả đạt được.

---

# 7_P07_data

Dữ liệu được dùng là NIST Roof, gồm công suất AC, bức xạ và nhiệt độ môi trường. Metadata công suất AC là 75 kilowatt. Dữ liệu gốc có chu kỳ một phút, được gộp thành trung bình năm phút khi đủ quan trắc; pipeline không nội suy để lấp những khoảng thiếu.

Dữ liệu được chia theo thời gian: năm 2015 dùng để train với 51.405 mẫu hợp lệ; năm 2016 dùng để validation với 47.487 mẫu; năm 2017 dùng để test với 43.806 mẫu. Việc chia theo năm phản ánh tình huống dùng quá khứ để dự báo tương lai và tránh trộn ngẫu nhiên các timestamp lân cận giữa train và test.

Đồ thị phía dưới là 25 quan trắc thực tế của ngày 1 tháng 6 năm 2015, chỉ lấy từ train. Nếu phát hành dự báo lúc 12 giờ, input chỉ được sử dụng thông tin có sẵn đến thời điểm đó. Phần nét đứt là quan trắc xảy ra về sau, được hiện lại để giải thích target; đó không phải đường dự báo.

Target trong ví dụ là công suất trung bình của bin từ sau 12 giờ 10 đến hết 12 giờ 15, có giá trị đo 51,71 kilowatt. Đây không phải trung bình của toàn bộ 15 phút tiếp theo. Giờ nguồn dùng EST cố định rồi được chuyển UTC. Độ trễ dữ liệu bằng không là giả định của bài kiểm tra offline, chưa phải độ trễ đã đo khi vận hành thực tế.

Sau khi xác định cách tạo mẫu, cần kiểm tra dữ liệu còn đủ độ phủ để đánh giá hay không.

---

# 8_P08_audit

Biểu đồ này trình bày bằng chứng về độ sẵn có dữ liệu. Hai dãy cột trả lời hai câu hỏi khác nhau.

Canonical daylight coverage cho biết độ phủ của dữ liệu canonical trên các mốc ban ngày dự kiến. Eligible coverage xét thêm các điều kiện để xây dựng được mẫu, như đủ lịch sử, target và tính liên tục. Vì vậy, eligible coverage thấp hơn canonical coverage là điều có thể xảy ra ngay cả khi dữ liệu nguồn có độ phủ cao.

Train có hai tỷ lệ khoảng 98,9% và 96,6%; validation là 97,3% và 89,0%; test là 97,6% và 82,3%. Cả ba split đều vượt ngưỡng canonical 90% và eligible 80%. Test còn có 157 ngày usable và 35 tuần usable không chồng lấn, vượt yêu cầu tối thiểu 150 ngày và 20 tuần. Audit được xác thực ngày 16 tháng 10 ghi nhận PASS.

Cần phân biệt coverage ở đây là độ sẵn có dữ liệu, còn coverage ở slide đánh giá là tỷ lệ quan trắc nằm trong khoảng dự báo. Audit cũng có giới hạn: do thiếu complete issue log, dự án dùng lịch sử thiết bị và QC định lượng theo amendment, nên vẫn có thể còn lỗi chưa gắn nhãn.

Kết quả này cho phép tiếp tục phép thử trên dữ liệu đã chuẩn bị; hiệu quả dự báo và khả năng suy rộng sang site khác vẫn phải được đánh giá riêng.

---

# 9_P09_status

Theo bằng chứng thực thi ngày 16 tháng 10, dự án đã hoàn thành pipeline và kiểm tra end-to-end trên dữ liệu tổng hợp, chuẩn bị dữ liệu NIST thật, audit độ sẵn có, xây dựng các split theo thời gian và profiling trên train.

Những việc này giúp kiểm tra luồng xử lý và lập kế hoạch chạy. Tuy nhiên, full training và đánh giá hiệu năng holdout chưa được thực hiện ở mốc bằng chứng đó, nên chưa có bảng kết quả mô hình thật để kết luận nhánh nào tốt hơn.

Nút thắt được ghi nhận là tài nguyên. Kế hoạch ngoại suy toàn benchmark stage khoảng 111 giờ và peak RAM khoảng 33,4 GiB; riêng baseline stage được ước lượng khoảng 72 giờ. Đây là các ước lượng từ profiling có hệ số dự phòng, có thể đếm chồng một số allocation. Chúng không phải thời gian full run đã đo, cũng không phải độ trễ của một lần dự báo hay tốc độ phần cứng lượng tử.

Vì vậy, hành động tiếp theo trong hồ sơ thực thi là xác minh cloud capacity trước khi chạy toàn bộ benchmark. Cần giải quyết đủ tài nguyên cho phép so sánh đã thiết kế, sau đó mới diễn giải kết quả khoa học.

---

# 10_P10_next

Kế hoạch tiếp theo gồm ba bước. Bước một là xác minh môi trường có đủ tài nguyên để chạy trong giới hạn đã thống nhất. Bước hai là chạy các baseline, mô hình photonic và đối chứng theo protocol; chọn siêu tham số bằng train và validation rồi khóa lựa chọn. Bước ba mới đánh giá holdout năm 2017 sau khi quá trình lựa chọn hoàn tất.

Quyết định GO, EXTEND hay STOP sẽ dựa đồng thời trên WIS, calibration và tài nguyên. GO khi lợi ích đủ rõ và đạt các gate. EXTEND khi bằng chứng chưa phân định được và có lý do cụ thể để thử thêm. STOP khi lợi ích không đạt hoặc chi phí không phù hợp.

Nếu mô hình chỉ tốt hơn photon phân biệt được nhưng vẫn kém baseline cổ điển mạnh, kết quả đó có thể gợi ý đóng góp của giao thoa nhưng chưa đủ lập luận cho ứng dụng. Nếu kết quả exact đủ thuyết phục, nhóm mới chuyển sang finite shots, mất photon, rồi cân nhắc xác minh phần cứng khi có cơ sở.

Điều pilot cần tạo ra là một câu trả lời đáng tin về vai trò của thành phần lượng tử trong pipeline. Ở mốc báo cáo này, dữ liệu và thiết kế phép thử đã được chuẩn bị; bước cần hoàn thành là benchmark công bằng để xác định thành phần lượng tử có cải thiện dự báo và có xứng đáng với chi phí hay không.
