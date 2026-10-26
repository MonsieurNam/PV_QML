<!-- ppt-master-schema: design-spec/v1 -->
# Photonic PV Overview - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | photonic_pv_overview_10slides_20261020 |
| Canvas Format | PPT 16:9, 1280 × 720 |
| Page Count | 10 |
| Primary Language | en-US for visible slide content; vi-VN for speaker notes and live delivery |
| Target Audience | Anh Quang và nhóm nghiên cứu/Quanova; có nền kỹ thuật nhưng không cần chuyên sâu photonic quantum computing. |
| Communication Intent | Báo cáo ngắn gọn trong tối đa 10 slide: giải thích động cơ, các nhánh và nghiên cứu đã khảo sát, lý do chọn PV, giả thuyết, phương pháp và dữ liệu; sau đó làm rõ tiến độ thực tế và bước thực hiện kế tiếp. |
| Desired Audience Outcome | Người nghe hiểu vì sao chọn photonic PV forecasting, cách kiểm tra giá trị của thành phần lượng tử, dữ liệu nào đã có, bằng chứng nào còn thiếu và nút thắt cần giải quyết trước full benchmark. |
| Core Message / Ask / Action | Dự án đã có pipeline, dữ liệu NIST 2015–2017 và audit PASS; bước tiếp theo là giải quyết tài nguyên để đánh giá công bằng photonic features. Chưa có kết quả mô hình thật hay bằng chứng quantum advantage. |
| Delivery Context | Chủ yếu thuyết trình trực tiếp khoảng 8–10 phút bằng tiếng Việt; sau buổi họp có thể đọc lại qua slide và speaker notes. Nội dung slide dùng tiếng Anh theo chỉnh sửa rõ ràng của người dùng. |
| Artifact Afterlife | PPTX chỉnh sửa được, lưu làm báo cáo tiến độ; speaker notes giữ giải thích và nguồn để tránh nhầm đề xuất với kết quả. |
| Reading Mode | presentation |
| Content Strategy | Balanced default; reorganize and distill supplied evidence without inventing facts. |
| Design Style | chalkboard with the confirmed navy, teal and amber palette; a navy board inset on the confirmed light canvas, with chalk-like teaching diagrams and Arial text. |
| AI Image Acquisition Path | Not applicable; user's later request authorizes an official site photograph and a deterministic plot from supplied measured training data. |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — final Stage-2 proactive policy |
| Custom Animations | disabled — final Stage 2 false; default transitions remain permitted |
| Narration Audio | disabled — final Stage-2 proactive policy |
| Created Date | 2026-10-20 |

Language resolution: the retained primary-language field was vi-VN, but the user's explicit edit to Delivery Context and the final page-count/image notes require all visible slide content in English. That explicit requirement owns the exported slide language; Vietnamese remains the notes/presenter language. The final custom mode is the decision-centered narrative selected by the user, not the initially recommended direction. The later explicit user request to review and add real images supersedes the earlier image_usage none only for documentary imagery and measured-data examples. Bundled icons remain unused; no AI imagery or model-score illustration is introduced.

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 24 px outer canvas frame; generous content breathing room inside the board |
| Content Area | Interior of the inset board with readable titles, short labels and compact source captions |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode Behavior**: Organize the argument around decisions: industrial requirements, surveyed directions, PV selection, hypothesis, fair test, and conditions for further investment; use NIST evidence to distinguish data readiness from model validation.
- **Visual style**: chalkboard
- **Theme**: An explanatory research board; paths, brackets and slightly irregular emphasis lines connect the decisions. Navy provides the slate field inside a light frame; pale chalk tints preserve readable accents.
- **Tone**: Precise, instructional, candid about uncertainty.

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #F7F9FC | Outer canvas and primary chalk on the navy board |
| Secondary background | #E7EDF4 | Secondary chalk and supporting text |
| Primary | #14263D | Navy slate field and dark ink when on light backgrounds |
| Accent | #008F86 | Teal feature-map identity and restrained fills |
| Secondary accent | #B56A00 | Amber stage/protocol identity |
| Body text | #223348 | Dark text on light evidence strips |
| Chalk teal | #80C7C2 | Readable teal tint on the navy board |
| Chalk amber | #E1BA82 | Readable amber tint on the navy board |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | Clear sans serif, bold; chalk character expressed by surrounding marks | Arial | Arial | sans-serif |
| Body | Legible sans serif | Arial | Arial | sans-serif |

- **Title stack**: Arial
- **Body stack**: Arial

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 28 |
| Title | 50 |
| Subtitle | 38 |
| Annotation | 22 |
| Footnote | 16 |
| Data | 58 |
| Hero | 72 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: One assertion first, one central explanation, one concise implication.
- **Composition tendency**: Teaching-board diagrams, open causal paths, paired controls, an authentic audit chart, and a lower-density closing decision.
- **Cross-page continuity**: Reuse the board, English section labels, source-caption convention and teal quantum/amber pending semantics. Vary diagram topology with the actual argument.
- **Spacing posture**: Open presentation pacing; denser literature/control evidence stays in notes when needed.

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P08 | chart | grouped_bar_chart | Compare canonical and eligible daylight coverage across chronological splits, using values from audit_coverage.csv. |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nist_roof.jpg | 960 × 640 | 1.5 | Identify the physical NIST Roof Tilted Array on P02 | Photography | Documentary evidence adjacent to the industrial motivation; show the complete scene with its source caption | no-crop | web | Sourced | Exact Roof Tilted Array photograph from the official NIST page; source URL and reuse-policy review recorded in image_sources.json | none | local |
| nist_train_example.png | 3468 × 1046 | 3.3155 | Show actual AC observations and the 15-minute target semantics on P07 | Scientific plot | Large complete plot below a compact chronological split strip; preserve every axis and qualifier | no-crop | user | Existing | Deterministic CNSPlots figure from supplied canonical data, fixed training date 2015-06-01 11:00–13:00 fixed EST; observed data only; scripts and CSV retained | none | local |

## IX. Content Outline

### Part 1: Why this project

#### Slide 01 - A testable photonic PV pilot

- **Audience move**: An unfamiliar project becomes a concrete research question with an honest current state.
- **Layout**: Large project statement beside a simplified PV-to-features-to-forecast path; one prominent status line.
- **Title**: Photonic PV forecasting
- **Core message**: From an industry-first question to a testable pilot.
- **Content**: Project overview; 20 October 2026. Can quantum optical features improve 15-minute-ahead probabilistic PV forecasts? Data and pipeline ready; full benchmark pending. Exact simulator study; no hardware advantage claim.
- **Visualization**: Qualitative three-stage PV history / photonic features / forecast path; no invented forecast curve.
- **Cover impact**: Bind the tension “Data ready. Evidence still to earn.” to a strong teaching-board composition.

#### Slide 02 - Start with the use case

- **Audience move**: A technology-led idea becomes a constrained industrial research program.
- **Layout**: Three requirements on the left; a documentary photograph of the selected NIST Roof installation on the right grounds the application in a physical system.
- **Title**: Start with a useful question
- **Core message**: Industry relevance and scientific contribution must both be testable.
- **Content**: Industrial relevance: measurable target and traceable data. Scientific value: a falsifiable role for quantum features. Feasibility: small simulation before QPU investment. Initial candidates: finance, equipment sensing, molecular/material learning and time series. No QPU access assumed.
- **Visualization**: Requirements remain qualitative and separately labeled; candidate domains remain a compact caption.
- **Images**: nist_roof.jpg, complete scene with NIST source caption; no implication that Quanova owns the installation or has operated quantum hardware.

#### Slide 03 - Literature shaped the design

- **Audience move**: Prior work becomes a set of concrete design constraints rather than a list of fashionable topics.
- **Layout**: Three evidence tracks with short paper citations leading to a design lesson.
- **Title**: Prior work changed the design
- **Core message**: The contribution must survive fair controls and realistic resource accounting.
- **Content**: Benchmarking: Bowles et al. (2024) -> strong classical controls. Mechanism: photonic kernel experiment (2025), Di Bartolo et al. (2026) -> isolate interference and memory. Resources/use case: Ahmed et al. (2025), Mills–Mezher (2026), Sagingalieva et al. (2025) -> account for shots/loss; QML for PV already exists. Damping, topology and weak measurement also have prior art. Targeted review; novelty is not yet established.
- **Visualization**: Qualitative paper-to-lesson tracks, not a ranking or numeric evidence score.

#### Slide 04 - Why PV and what to test

- **Audience move**: The choice of PV becomes a justified test case with a falsifiable hypothesis.
- **Layout**: A large research question supported by three concise reasons and a bounded current scope.
- **Title**: Why PV is a useful first test
- **Core message**: Public measured data and clear baselines make the hypothesis testable.
- **Content**: Measured power + weather. Clear 15-minute target. Strong probabilistic baselines. Hypothesis: indistinguishable-photon features improve forecast quality beyond classical features and distinguishable controls. Current scope: exact, 2 photons, 4 modes. Finite shots, loss and site shift are later tests.
- **Visualization**: Qualitative selection reasons and a chalk-ring emphasis on the hypothesis; no claim of confirmed novelty.

### Part 2: How it will be tested

#### Slide 05 - Model architecture

- **Audience move**: Trace inputs through the proposed architecture and comparator branches to a common forecast target and evaluation metrics.
- **Layout**: Input/preprocessing above three model branches, with explicit branching and convergence arrows leading to one output block; metrics in the lower strip.
- **Title**: Proposed model and comparator paths
- **Core message**: Explicit inputs, common targets and registered metrics support the controlled comparison; memory remains classical.
- **Content**: Matched input P(t), P(t−15), GHI(t), T(t); train-only scaling. Proposed exact model: Fock input -> fixed V1 -> data-dependent D(phi(x)) -> fixed V2 -> 10 probabilities -> trained classical quantile head; two photons / four modes / five seeds. Mechanism control: distinguishable photons, same circuit, inputs and readout policy. Classical comparators: raw inputs or 10-dimensional RFF -> quantile head; quantile XGBoost; persistence. XGBoost explicitly uses power lags [0,5,10,15,30,60] plus current weather and solar geometry. Output: seven power quantiles [0.05,0.10,0.25,0.50,0.75,0.90,0.95] in kW for the 5-minute mean ending at t+15; q50 and [q05,q95] summarize the same output. Metrics: normalized WIS and relative system skill, 90% coverage and normalized interval width, runtime and peak RAM. Planned comparisons, not model results.
- **Visualization**: Editable qualitative architecture with explicit branch/convergence arrows; phase between two fixed mixing layers. No invented probability histogram or forecast curve. User-requested expansion on 23 October 2026 replaces the former P05 and retains 10 slides.

#### Slide 06 - Fair comparison

- **Audience move**: The research claim becomes separable into application benefit and interference contribution.
- **Layout**: Two branches of one controlled comparison, followed by a concise gate statement.
- **Title**: Two comparisons, two different claims
- **Core message**: Beating an application baseline and isolating interference require different controls.
- **Content**: Application: persistence, raw + quantile head, RFF + same head, strong quantile XGBoost. Mechanism: indistinguishable versus distinguishable photons; same inputs, circuit and readout policy. Train/validation select; freeze before 2017. Primary metric: capacity-normalized WIS, lower is better. Relative system skill = 1 − WIS(system)/WIS(B), with B selected on validation. Calibration: 90% empirical coverage and mean capacity-normalized interval width. Resources: wall-clock runtime and peak RAM. Paired day-block bootstrap supplies 95% confidence intervals. GO: lower 95% CI of relative system WIS skill above 2%, coverage within 85–95%, and resource gates pass. Criteria, not results.
- **Visualization**: Qualitative branches and a shared evaluation endpoint, not a pure cell grid.

#### Slide 07 - Data and target

- **Audience move**: Dataset identity, target semantics and split boundaries become unambiguous.
- **Layout**: Compact chronological split strip above a full-width measured training-data trace; show target semantics directly on the trace.
- **Title**: One site, three chronological years
- **Core message**: NIST Roof 2015–2017 provides a reproducible offline test.
- **Content**: Measured AC power + irradiance + ambient temperature; AC metadata 75 kW. 1-minute source -> complete 5-minute means, no interpolation. Train 2015: 51,405 samples. Validation 2016: 47,487. Test 2017: 43,806. Observed training example on 1 June 2015: issue 12:00 -> target mean 12:10–12:15, measured 51.71 kW. Fixed EST -> UTC. Zero availability delay is an offline assumption; sensor is a horizontal silicon reference cell. The observed trace is not a forecast.
- **Visualization**: Chronological splits use labels rather than value-driven geometry. The prepared scientific plot shows real power bins over 11:00–13:00 on 2015-06-01, an issue marker at 12:00 and the target interval 12:10–12:15. It is an observed-data example, not a forecast or model result.
- **Images**: nist_train_example.png displayed complete with axes and qualifier; source CSV and plotting code remain in sources/ and scripts/ for reproduction.

#### Slide 08 - Audit evidence

- **Audience move**: Data readiness is understood as measured coverage, separate from forecast quality.
- **Layout**: Large grouped coverage chart and a compact test-support summary.
- **Title**: Data readiness passed its audit
- **Core message**: Registered availability gates pass; forecast quality has not been evaluated.
- **Content**: Canonical/eligible daylight coverage: train 98.9%/96.6%; validation 97.3%/89.0%; test 97.6%/82.3%. Canonical threshold 90%; eligible threshold 80%. 2017 support: 157 usable days; 35 usable non-overlapping weeks. Coverage here means data availability, not prediction-interval coverage. Audit PASS, 16 Oct 2026.
- **Visualization**: coverage-chart = grouped vertical bars, categories Train/Validation/Test, two series, common 0–100% axis; source audit_coverage.csv. Test support figures are text labels, not mini charts.
- **Native-ready**: coverage-chart=yes

### Part 3: What happens next

#### Slide 09 - Current status and constraint

- **Audience move**: Software/data progress is distinguished from the unresolved full-run capacity decision.
- **Layout**: Completed path on one side and one prominent capacity constraint on the other.
- **Title**: Ready to test; capacity is the blocker
- **Core message**: The latest run prepared and profiled the study but did not run the full benchmark.
- **Content**: Done: pipeline and synthetic E2E; NIST preparation and audit; chronological splits; train-only profiling. Not run: full model comparison and holdout performance. Benchmark projection: 33.4 GiB peak RAM and about 111 hours for the whole stage. Baseline stage projection: about 72 hours. Profiling extrapolations, not observed full-run timings. Latest route: local-failed / cloud-verification-required. Evidence run: 16 Oct 2026.
- **Visualization**: Qualitative completion sequence and explicit text-only resource estimates; no runtime bars or fabricated model-score chart.

#### Slide 10 - Next experiment and decision

- **Audience move**: The audience knows the next concrete action and what evidence would justify further investment.
- **Layout**: A short sequential decision path ending with a large takeaway.
- **Title**: Earn the next stage with evidence
- **Core message**: Resolve compute capacity, run the frozen comparison, then decide.
- **Content**: 1 Verify capacity. 2 Run locked baselines and photonic controls. 3 Evaluate 2017 once selections are frozen. GO / EXTEND / STOP using WIS, calibration and resources. Only then consider finite shots/loss and hardware validation. Takeaway: test whether the quantum component earns its place.
- **Visualization**: Qualitative three-step path with later stages visually subordinated.
- **Closing impact**: Bind “Test whether the quantum component earns its place” to the final decision rather than a generic thank-you page.

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: Match P01–P10 SVG basenames.
- **Content**: Vietnamese presenter notes; explain each final visual, transitions, full or abbreviated paper titles and local evidence paths. Distinguish adopted protocol criteria, historical software checks, measured audit facts and resource extrapolations. No invented model scores, hardware performance, confirmed novelty or user approval.
- **Total duration**: 8–10 minutes.
- **Notes style**: Clear research briefing, conversational but precise.
- **Presentation purpose**: Explain the research choices, report progress and align the next experiment.
