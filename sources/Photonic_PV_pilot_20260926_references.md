# Nguồn cho đánh giá pilot photonic PV — 26/09/2026

Danh mục có mục tiêu, không phải thống kê toàn bộ field. DOI được kiểm tra riêng; DOI hợp lệ không đồng nghĩa kết quả đã được tái lập. Không suy ra Q1 từ thương hiệu tạp chí.

## Nguồn được dẫn trong báo cáo

1. Ahmed, O., Tennie, F., & Magri, L. (2025). Optimal training of finitely sampled quantum reservoir computers for forecasting of chaotic dynamics. *Quantum Machine Intelligence, 7*, 31. https://doi.org/10.1007/s42484-025-00261-9
   - Peer-reviewed. Đối chiếu trang Springer và bản lưu tác giả tại https://openaccess.city.ac.uk/id/eprint/34976/ ; đã kiểm tra abstract và nội dung liên quan finite sampling, không tái lập kết quả.

2. Di Bartolo, R., et al. (2026). Time-series forecasting with multiphoton quantum states and integrated photonics. *npj Quantum Information, 12*, 91. https://doi.org/10.1038/s41534-026-01236-9
   - Peer-reviewed, xuất bản 17/04/2026; đọc phần protocol, experiment và discussion. Thiết bị nghiên cứu riêng, không phải bằng chứng trực tiếp về cloud workflow của Quanova.

3. Mills, J., & Mezher, R. (2026). Mitigating photon loss in linear optical quantum circuits. *Quantum, 10*, 2030. https://doi.org/10.22331/q-2026-03-16-2030
   - Peer-reviewed, xuất bản 16/03/2026; đối chiếu https://quantum-journal.org/papers/q-2026-03-16-2030/ và https://arxiv.org/abs/2405.02278 . Đọc abstract, mô tả phương pháp và phạm vi kết quả; không tái lập thuật toán.

4. López Carreño, J. C., et al. (2026). Quantum and classical processing with photonic quantum machine learning. *arXiv*, 2605.10471. https://arxiv.org/abs/2605.10471
   - Preprint nộp 11/05/2026; đọc HTML, đặc biệt mitigation/experimental details. Liên hệ Quandela được xác nhận ở trang publication và tác giả. Không coi là peer-reviewed journal paper khi chưa xác minh được trạng thái đó.

5. Cimini, V., et al. (2026). Large-scale quantum reservoir computing using a Gaussian Boson Sampler. *npj Quantum Information, 12*, 104. https://doi.org/10.1038/s41534-026-01251-w
   - Peer-reviewed; đọc introduction và giới hạn quantum-advantage claim trên trang publisher.

6. Sagingalieva, A., et al. (2025). Photovoltaic power forecasting using quantum machine learning. *Solar Energy, 302*, 114016. https://doi.org/10.1016/j.solener.2025.114016
   - Peer-reviewed record xác nhận qua https://arxiv.org/abs/2312.16379 ; publisher toàn văn không truy cập được. Không dùng phần trăm cải thiện trong bài làm dự báo hiệu quả cho pilot này.

7. Liu, W., et al. (2026). A two-stage hybrid framework for photovoltaic power forecasting using MTF-MTS-Mixer and physics-informed Bayesian neural networks. *Renewable Energy, 270*, 125960. https://doi.org/10.1016/j.renene.2026.125960
   - Publisher abstract/highlights, số tạp chí 15/08/2026. Chỉ dùng chứng minh tồn tại hướng classical probabilistic/physics-informed, chưa thẩm định toàn bộ thiết kế.

8. Chen, K., Li, Y., & Wang, J. (2026). Towards reliable photovoltaic power interval forecasting: A novel cross-channel multi-scale fusion transformer with non-crossing multi-quantile learning. *Information Sciences, 748*, 123505. https://doi.org/10.1016/j.ins.2026.123505
   - Publisher abstract, số tạp chí 25/08/2026. Chưa kiểm tra toàn văn hoặc code, không xác nhận tuyên bố chống leakage của tác giả.

9. Huang, C., Ma, Z., & Michailidis, G. (2026). Model-Agnostic Online Certificate-Driven Calibration for Time Series Forecasting Under Distribution Shift. *UAI / PMLR, 337*, 2244–2273. https://proceedings.mlr.press/v337/huang26b.html
   - Kỷ yếu hội nghị tháng 8/2026; kiểm tra abstract và metadata. Không phải bằng chứng trực tiếp về calibration trên PV hoặc quantum features.

10. Amanov, F., & Azamov, A. (2026). Hybrid Photonic Quantum Reservoir Computing for High-Dimensional Financial Surface Prediction. *arXiv*, 2603.10707. https://arxiv.org/abs/2603.10707
    - Preprint; đọc kiến trúc và mục 7.4 Limitations. Không sử dụng tốc độ được báo cáo làm latency QPU E2E.

11. Rambach, M., et al. (2025). Photonic Quantum-Accelerated Machine Learning. *arXiv*, 2512.08318. https://arxiv.org/abs/2512.08318
    - Đọc preprint; có bản trình bày Quantum 2.0, 15–18/06/2026, QW4A.8: https://opg.optica.org/abstract.cfm?uri=QUANTUM-2026-QW4A.8 . Không đánh đồng conference digest với journal validation của full paper.

12. XGBoost. Quantile Regression (tài liệu chính thức). https://xgboost.readthedocs.io/en/release_2.1.0/python/examples/quantile_regression.html
    - Xác nhận objective quantile và cảnh báo crossing; khi triển khai cần khóa version thực sự cài đặt.

13. Zaffran, M., et al. (2022). Adaptive Conformal Predictions for Time Series. *ICML / PMLR, 162*, 25834–25866. https://proceedings.mlr.press/v162/zaffran22a.html
    - Nguồn phương pháp, không phải nghiên cứu photonic forecasting. Bổ sung Gibbs–Candès (2024), *JMLR*: https://jmlr.org/papers/v25/22-1218.html .

14. EPRI / Solar Forecast Arbiter. Benchmark Forecasts. https://forecastarbiter.epri.com/benchmarks/
    - Tài liệu chính thức về persistence, clear-sky persistence, NWP và probabilistic baselines.

15. Perceval v1.2. NoisySamplingSimulator. https://perceval.quandela.net/docs/v1.2/reference/simulators/noisy_sampling_simulator.html
    - `max_samples` khác `max_shots`; physical/logical filtering phải tính vào resource accounting.

16. Perceval v1.2. Processor. https://perceval.quandela.net/docs/v1.2/reference/runtime/processor.html
    - Noise model, input state và min-photon filter. Danh mục noise chi tiết cũng đối chiếu tài liệu cũ, có ghi version: https://perceval.quandela.net/docs/v0.13/reference/noise_model.html . Không trộn API các version khi viết code.

17. NREL. Solar Power Data for Integration Studies. https://www.nrel.gov/grid/solar-power-data.html?print=
    - Nguồn chính thức: dữ liệu synthetic 5 phút; không mặc định “NREL” đồng nghĩa công suất nhà máy được đo.

18. Quandela (22/06/2026). Real-Time GPU–QPU Integration for Photonic Quantum Machine Learning using NVQLink. https://www.quandela.com/resources/blog/quandela-nvidia-gpu-qpu-integration-nvqlink/
    - Nguồn nhà cung cấp; không phải phép đo hệ thống Quanova, không xác nhận quyền dùng công nghệ.

## Bổ sung để mở rộng đọc, chưa dùng xác lập tính mới

19. Od, M., et al. (2026). Quantum Reservoir Computing for Short-Term Power Load Forecasting in Resource-Constrained Energy Systems. *arXiv*, 2606.12806. https://arxiv.org/abs/2606.12806
    - Preprint, đọc abstract. Load forecasting không đồng nhất PV power forecasting; noise simulation không phải chạy IBM thật.

20. Tariq, S., et al. (2026). Quantum Reservoir Computing: Recent Advances and Future Directions. *arXiv*, 2607.18552. https://arxiv.org/abs/2607.18552
    - Survey preprint ngày 20/07/2026, dùng định hướng tìm kiếm, không lấy làm thay thế nguồn thực nghiệm gốc.

21. Paparelle, I., et al. (2026). Experimental memory control in continuous-variable optical quantum reservoir computing. *Nature Photonics, 20*, 413–420. https://doi.org/10.1038/s41566-026-01880-9
    - Abstract đã kiểm tra. Nền tảng CV có memory được thiết kế riêng, không phải khả năng đã xác nhận của cloud Quandela.

## Nhật ký giới hạn

- Semantic Scholar API: HTTP 429. Không giả lập số kết quả hoặc PRISMA flow.
- Crossref query `photonic quantum probabilistic solar forecasting`, rows=5, cutoff 26/09/2026: top metadata có cả quantum-inspired/SSRN; không lấy làm bằng chứng triển khai quantum và không dùng để tuyên bố không có prior art.
- Một số URL publisher trả lỗi truy cập; trạng thái đọc từng nguồn ở trên phân biệt abstract với các phần toàn văn.
- Chưa kiểm tra toàn bộ supplementary material, chưa liên hệ tác giả, chưa kiểm tra code/data toàn diện.
