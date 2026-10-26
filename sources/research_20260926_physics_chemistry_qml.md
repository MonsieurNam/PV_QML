# QML for industrial physics, chemistry and materials

Date: 2026-09-26. Targeted extension of the industry-first landscape; not a systematic review. No code executed or QPU experiments performed. Research-lookup method used with built-in web fallback because its configured API backends were unavailable in the preceding environment check. Table/prose is sufficient; no generated schematic needed.

## Scope distinctions

Industrial relevance includes materials screening, chemical formulation, drug discovery, engineering models and measurement systems. QML using quantum processors is distinct from classical ML trained on DFT data and from quantum simulation/eigensolvers without a learning task. A molecular target does not automatically make a classical-descriptor prediction problem quantum-advantaged.

## Primary-source notes

1. Beaulieu et al., *Robust Quantum Reservoir Computing for Molecular Property Prediction*, arXiv v1, 2024-12-09.
   https://arxiv.org/abs/2412.06758
   https://arxiv.org/html/2412.06758v1
   https://doi.org/10.48550/arXiv.2412.06758
   Related journal DOI listed by arXiv: https://doi.org/10.1021/acs.jcim.5c00958 (publisher access blocked; inspected preprint).
   Collaborators include Deloitte, QuEra, Amgen and Merck KGaA. Uses Bloqade simulation of neutral-atom reservoirs on molecular-activity descriptors, reporting resilience to reduced training sizes in the studied comparisons. Not a photonic hardware demonstration. Merck KGaA collaboration and historical Merck & Co. challenge data are distinct entities.

2. Ghazi Vakili et al., *Quantum-computing-enhanced algorithm unveils potential KRAS inhibitors*, Nature Biotechnology, online 2025-01-22.
   https://www.nature.com/articles/s41587-024-02526-3
   https://doi.org/10.1038/s41587-024-02526-3
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12700792/
   Hybrid QCBM/LSTM molecular generation; 15 molecules synthesized and experimentally tested, with two highlighted for further development. Quantum hardware sampling is part of the workflow. Early biological hits are not approved drugs or proof of generally superior quantum discovery. Full-page access was intermittent; primary publisher indexed text provided abstract and workflow details.

3. Kang and Kim, *Property-guided inverse design of metal-organic frameworks using quantum natural language processing*, npj Computational Materials, 2025.
   https://www.nature.com/articles/s41524-025-01806-z
   https://doi.org/10.1038/s41524-025-01806-z
   Explores 450 hypothetical MOFs and pore-volume / CO2 Henry-constant targets with Qiskit classical simulation. Authors explicitly distinguish their feasibility study from demonstrating quantum computational advantage. Material selection in a constrained component space is not unrestricted chemical generation or experimentally validated CO2 capture.

4. Markazi and Mirza, *Evaluating the performance of classical, quantum, and hybrid quantum neural networks in solving differential equations*, Scientific Reports, 2025-11-10.
   https://www.nature.com/articles/s41598-025-22950-y
   https://doi.org/10.1038/s41598-025-22950-y
   Small equation benchmarks and initialization-sensitive comparisons. Do not infer superiority to production FEM/CFD. Real-exponential nonunitary feature operations in the presented formulation require implementation scrutiny before transferring to hardware; this register does not endorse its resource claims.

5. Remond et al., *Quantum remeshing and efficient encoding for fracture mechanics*, 2025-10-16.
   https://arxiv.org/abs/2510.14746
   https://doi.org/10.48550/arXiv.2510.14746
   https://www.quandela.com/resources/blog/quantum-fracture-mechanics-quantum-processor/
   Quandela/EDF context: variational energy minimization for fracture mechanics with Ascella validation. Relevant industrial physics, but a quantum solver rather than automatically a QML model.

6. *A versatile single-photon-based quantum computing platform*, Nature Photonics, 2024.
   https://www.nature.com/articles/s41566-024-01403-4
   https://doi.org/10.1038/s41566-024-01403-4
   https://github.com/Quandela/Ascella
   Platform demonstrations and notebooks include VQE and classification separately. A VQE demo is not evidence of industrial chemical-learning advantage.

7. *The QCML dataset, Quantum chemistry reference data from 33.5M DFT and 14.7B semi-empirical calculations*, Scientific Data, 2025.
   https://www.nature.com/articles/s41597-025-04720-7
   https://doi.org/10.1038/s41597-025-04720-7
   Quantum-chemistry reference labels for ML, illustrating why the word quantum does not necessarily imply a quantum processor.

8. *MOFSimBench: evaluating universal machine learning interatomic potentials in metal-organic framework molecular modeling*, online 2025-12-11, volume 2026.
   https://www.nature.com/articles/s41524-025-01872-3
   https://doi.org/10.1038/s41524-025-01872-3
   Classical ML baseline context: practical evaluation includes structural relaxation, dynamics, bulk properties and host-guest interactions, not only a single prediction error.

## Implications — proposals, not established findings

- Include physics, chemistry and materials in the industry shortlist. They should not be omitted merely because earlier examples were finance/cybersecurity.
- Candidate A: sample-efficient prediction/ranking for a specific molecular or material property, evaluated under scaffold/composition-family splits and comparable classical baselines. Active-learning savings should include label acquisition and descriptor costs.
- Candidate B: physics-informed inverse estimation or a surrogate for a narrowly specified physical system. Compare with conventional numerical inversion, Gaussian processes and relevant classical scientific ML. A toy differential equation alone is insufficient industrial evidence.
- Quantum-native state/property learning is relevant to quantum-device characterization but requires a credible route to quantum data; it is a different access requirement from a CSV of computed molecular properties.
- No-QPU phase: validate data provenance, reproduce classical references, simulate small quantum models and estimate shot/noise costs. Mapping neutral-atom or superconducting architectures to Quandela needs explicit redesign and validation.
- Do not assume linear photonics natively simulates electronic molecular Hamiltonians, and do not claim materials/chemistry performance leadership by Quandela without task-specific evidence.
- Select one measurable industrial target and a domain collaborator before promoting any of these candidates to a project commitment.
