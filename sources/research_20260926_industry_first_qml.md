# Industry-first QML: focused research update

Date: 2026-09-26. User clarification: investigate QML for industry; do not preselect a generative algorithm merely because Quandela has a recent paper. Scope is learning applications, not quantum optimization, quantum chemistry simulation or post-quantum cryptography unless explicitly part of an ML workflow.

Method: targeted public-source scan, not systematic coverage. Reused the companion landscape register. Research-lookup backend keys are unavailable; used built-in web search and primary sources as fallback. No experiments run. No QPU access assumed. Diagrams omitted because the case comparison is adequately represented as a table in the response.

## Concrete cases and evidence

### 1. Finance: bond trade fill-probability estimation

- Primary technical source: Ciceri et al., *Enhanced fill probability estimates in institutional algorithmic bond trading using statistical learning algorithms with quantum computers*, 2025-09-22, version 1 inspected.
- https://arxiv.org/abs/2509.17715
- https://arxiv.org/html/2509.17715v1
- DOI: https://doi.org/10.48550/arXiv.2509.17715
- HSBC announcement, 2025-09-25: https://www.hsbc.com/news-and-views/news/media-releases/2025/hsbc-demonstrates-worlds-first-known-quantum-enabled-algorithmic-trading-with-ibm
- Real production-scale historical data and IBM Heron experiments; quantum transforms are an offline feature component. Authors report relative test-score gains and a potential role for device noise. This is a backtesting study, not evidence of universally superior ML or demonstrated live trading profit. Private data and platform differences constrain reproduction at Quanova.

### 2. Finance: credit-default prediction

- Credit Agricole CIB / Quandela announcement, 2025-10-09.
- https://presse.credit-agricole.com/actualites/linnovation-quantique-au-service-de-la-finance-credit-agricole-cib-et-quandela-optimisent-les-modeles-de-prediction-des-risques-de-defaut-846a6-9ed05.html
- Partners report a hybrid predictive improvement validated on photonic QPUs. Public announcement lacks enough detail to independently audit superiority. Follow up on data availability, class imbalance, calibration and business decision metrics rather than assuming an accuracy gain establishes utility.

### 3. Cybersecurity: malware detection

- Quandela's public account describing its Orange collaboration:
- https://www.linkedin.com/posts/quandela_quandela-x-orange-malware-use-case-activity-7403784365840945152-3w2I
- Preliminary QCNN work plus a distinct QML model reported to match Orange's model accuracy and to have been tested on QPU. Do not merge the architectures or equate fewer parameters with less total computation. The source is a company post, not a full benchmark publication. Malware detection is not itself post-quantum cryptographic security.

### 4. Manufacturing/process equipment: vibration anomaly detection

- *Machine Failure Detection Based on Projected Quantum Models*, 2026-01-22; v1 inspected.
- https://arxiv.org/abs/2601.15641
- https://arxiv.org/html/2601.15641v1
- DOI: https://doi.org/10.48550/arXiv.2601.15641
- Uses vibration-derived sensor features from fin-fan coolers on propane condenser heat exchangers. Seven labeled sequences; daily aggregation reduces hardware burden. Projected quantum features are evaluated with a density-ratio anomaly detector on simulator and IBM hardware. Small evaluation scope and limited comparisons do not establish general advantage over industrial anomaly detection. Dataset access and photonic implementation feasibility remain to be checked.

### 5. Mobility: rare-scenario data generation

- BMW Group / Airbus, *BMW Group and Airbus reveal winners of Quantum Computing Challenge*, 2024-12-10.
- https://www.press.bmwgroup.com/global/article/detail/T0446782EN/bmw-group-
- Quandela won the QML category exploring critical-scene generation, including day-to-night transformation. Industrial challenge evidence, not proof of production deployment or independent quantum advantage. This application is distinct from Quandela's 2026 fixed-cardinality photonic generative paper; the two must not be conflated.

## Selection framework — proposed, not findings from the papers

Assess each candidate by: business owner and decision; available data and permissions; strong classical reference; measurable business-related target; falsifiable reason a quantum component may help; reproducible code/protocol; no-QPU first experiment; photonic mapping cost; resource/noise budget; independent validation; stop criteria.

Suggested initial shortlist for deeper comparison: sensor anomaly detection, credit-risk/rare-event classification, malware detection. This is not a ranking of demonstrated quantum advantage. Rare-scenario generation remains an alternative if Quanova has a relevant customer and dataset. Bond-trading work is useful as an industry-workflow example, but difficult to reproduce directly without partner data.

Do not choose the final topic until data access and business relevance are known. Public datasets can support initial methods work, but do not alone demonstrate product readiness. Simulator studies can reject weak ideas and estimate resources; they cannot establish measured hardware speed or energy gains.

Immediate deliverable: compare three candidate use cases on one page each, with source quality, dataset options, classical baselines, relevant metrics, quantum role, hardware requirements, novelty uncertainties and go/no-go criteria. The industry requirement comes before the algorithm choice.
