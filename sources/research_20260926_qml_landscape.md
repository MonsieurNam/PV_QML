# QML landscape and Quandela: source register

Research cutoff: 2026-09-26. Scope: public research, primary company announcements, public researcher discussion. This is a targeted landscape scan, not a systematic literature review or an audit of unpublished industrial results. No experiments or QPU executions were performed.

## Evidence rules

- Separate theoretical guarantees, numerical experiments, physical demonstrations, and partner announcements.
- Distinguish predictive quality, training cost, sampling cost, memory, data requirements, and end-to-end runtime.
- A hard-to-sample circuit does not by itself establish useful learning advantage. A faster simulator is not a QPU speedup.
- Discussion threads illustrate specific disagreements, not a representative survey of the community.
- Claims about Quandela concern its public work, not its complete internal roadmap.

## Field-level research

1. Bowles, Ahmed, Schuld. *Better than classical? The subtle art of benchmarking quantum machine learning models* (2024).
   https://arxiv.org/abs/2403.07059
   Empirical benchmark: 12 quantum models and 160 small classification datasets. Classical comparators generally perform better; some entanglement removals do not hurt performance. Scope is the tested models and datasets, not all QML.

2. Yamasaki, Isogai, Murao. *Advantage of quantum machine learning from general computational advantages* (2026-07-28).
   https://www.nature.com/articles/s41534-026-01279-y
   Theoretical constructions connect computational and learning advantages. Not an end-to-end industrial benchmark.

3. Molteni, Gyurik, Dunjko. *Exponential quantum advantages in learning quantum observables from classical data* (2026-01-10).
   https://www.nature.com/articles/s41534-025-01162-2
   Theoretical advantage for structured learning connected to quantum observables; classical input need not mean an arbitrary industrial table.

4. Saem, Tafreshi, Holmes, Thanasilp. *Pitfalls when tackling exponential concentration in quantum machine learning* (2025).
   https://arxiv.org/abs/2507.22054
   Measurement-resource limitations remain important when assessing proposed concentration remedies.

5. Recio-Armengol, Ahmed, Bowles. *Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits* (v2, 2026-02-06).
   https://arxiv.org/abs/2503.02934
   IQP-based classical training with a prospective quantum sampling stage. Large numerical model size must not be presented as a physical hardware demonstration or assumed native to Quandela.

## Quandela research and tools

6. Notton et al. *Establishing Baselines for Photonic Quantum Machine Learning: Insights from an Open, Collaborative Initiative* (2025-10-29).
   https://arxiv.org/abs/2510.25839
   https://arxiv.org/html/2510.25839v1
   https://github.com/Quandela/HybridAIQuantum-Challenge
   Collaborative reduced-MNIST benchmark explicitly reports no clear heuristic quantum advantage at this stage. Individual comparisons use different configurations and should not be combined as one leaderboard.

7. Notton et al. *MerLin: A Discovery Engine for Photonic and Hybrid Quantum Machine Learning* (v2, 2026-04-17; first submitted 2026-02-11).
   https://arxiv.org/abs/2602.11092
   https://arxiv.org/html/2602.11092v2
   https://github.com/merlinquantum/merlin
   https://github.com/merlinquantum/reproduced_papers
   Framework and reproduction work covering 18 studies: kernels, reservoirs, hybrid networks, generative learning and other approaches. Differentiable classical simulation is a development capability, not evidence of quantum advantage.

8. Gottlieb et al. *Efficient training of photonic quantum generative models* (v2, 2026-07-03; v1, 2026-03-09).
   https://arxiv.org/abs/2603.08793
   https://arxiv.org/html/2603.08793v2
   https://github.com/merlinquantum/generative_classical_training
   Quandela preprint: classically estimated MMD training of photonic generative models, targeting quantum sampling. Numerical scale reaches 256 modes and 16 photons, not a QPU run at that size. Kernel and initialization comparisons already appear in v2. RBM comparisons are not proof of superiority over strong modern generative baselines. Fixed-cardinality representations, collisions, and physical noise require care.

9. Lopez Carreno et al. *Quantum and classical processing with photonic quantum machine learning* (2026-05-11).
   https://arxiv.org/abs/2605.10471
   https://www.quandela.com/resources/blog/apprentissage-automatique-quantique-photonique/
   Photonic reservoir experiments address classical tasks and quantum-state properties. Comparison with the same device in a classical regime is not comparison with the best classical ML algorithms.

10. Quandela MerLin announcement (2026-02-19).
    https://www.quandela.com/resources/blog/merlin-photonic-quantum-machine-learning-engineered-for-utility/
    Primary company explanation of its reproducible hybrid-development approach; interpret as strategy alongside papers and code.

11. Quandela QPU/NVIDIA NVQLink integration announcement (2026-09-14).
    https://www.quandela.com/about-us/newsroom/qpu-integration-nvidia-nvqlink/
    Places QPU workloads inside CPU/GPU workflows. Architecture and integration direction, not a general industrial performance guarantee.

## Industry signals

12. Credit Agricole CIB and Quandela announcement (2025-10-09).
    https://presse.credit-agricole.com/actualites/linnovation-quantique-au-service-de-la-finance-credit-agricole-cib-et-quandela-optimisent-les-modeles-de-prediction-des-risques-de-defaut-846a6-9ed05.html
    Partners report improved credit-default prediction validated on photonic processors. The accessed announcement does not provide enough dataset, baseline, uncertainty, or end-to-end resource detail for an independent superiority assessment.

13. Quandela / Orange public use-case announcement.
    https://www.linkedin.com/posts/quandela_quandela-x-orange-malware-use-case-activity-7403784365840945152-3w2I
    Preliminary malware work includes a QCNN and another QML model. Do not conflate the models or infer every architecture was hardware-tested. Treat as partner/application signal, not independent benchmarking evidence.

## Community discussion

14. Zhao et al. *Exponential quantum advantage in processing massive classical data* (2026-04-08).
    https://arxiv.org/abs/2604.07639
    Quantum oracle sketching; the claimed streaming-space separation is not a generic runtime speedup.

15. Joseph Bowles, SciRate comments (2026-04-10 to 2026-04-13).
    https://scirate.com/joseph-bowles
    Requests concrete gate-time and problem-size estimates; acknowledges the theoretical space improvement.

16. Aram Harrow, SciRate comment (2026-04-10).
    https://scirate.com/aram/comments
    Clarifies that the discussed exponential improvement concerns space, not time.

17. Haimeng Zhao, SciRate author responses (2026-04-10 to 2026-04-13).
    https://scirate.com/haimeng-zhao/comments
    Explains the end-to-end theoretical space claim, distinguishes engineering throughput, and acknowledges architecture-specific practical estimates as further work. SciRate direct retrieval was intermittently blocked; relevant comment text was available through indexed search results.

## Proposed implication for Quanova — interpretation, not a literature result

Update following user clarification: the priority is industry-first QML research. The generative direction below remains one candidate, not the selected project. Compare business use cases, accessible data, evidence and hardware feasibility before selecting an algorithm. See research_20260926_industry_first_qml.md.

Prioritize a feasibility study of classically trained photonic generative models, conditional on finding industrial data with a natural fixed-cardinality meaning. Proposed question: can useful distribution quality survive realistic sampling and noise constraints while remaining competitive with strong classical generators?

Start with the authors' latest public implementation at a small exactly checkable scale. Benchmark empirical resampling, simple constrained sampling, RBM, and a constrained autoregressive model with matched tuning budgets. Evaluate both distributional quality and a concrete downstream task, across independent splits/seeds. Add finite-shot, collision/postselection and hardware-noise sensitivity; report accepted-shot cost as well as nominal shots. Do not claim energy savings without measurement or justified system-level estimates.

This is a candidate direction, not a confirmed novel contribution. Merely replacing a kernel is not a sufficient novelty claim: the July paper already studies kernels. Perform a focused novelty review before committing. If data fit or competitive quality fails, stop or reframe instead of forcing a quantum-win narrative. QPU access is needed later for physical validation, not to begin reproducibility and feasibility work now.
