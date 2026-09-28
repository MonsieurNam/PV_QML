"""Runtime compatibility checks for the pinned Perceval release."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import perceval as pcvl


@dataclass(frozen=True, slots=True)
class PercevalCompatibility:
    """Observed results from the Perceval compatibility probe."""

    backend_name: str
    backend_class: str
    full_distribution_size: int
    full_distribution_probability: float
    distinguishable_distribution_size: int
    distinguishable_distribution_probability: float


def verify_perceval_compatibility(seed: int = 0) -> PercevalCompatibility:
    """Exercise the Perceval 1.3 APIs required by later pilot milestones."""
    pcvl.random_seed(seed)

    backend = pcvl.BackendFactory.get_backend("SLOS")
    simulator = pcvl.Simulator(backend)
    simulator.set_circuit(pcvl.BS())

    full_distribution = simulator.probs(pcvl.BasicState([1, 0]))
    full_probability = float(sum(full_distribution.values()))
    if not full_distribution or not isclose(full_probability, 1.0, abs_tol=1e-12):
        raise RuntimeError("SLOS returned an invalid full probability distribution")

    # Perceval 1.3 represents distinguishability tags with NoisyFockState syntax.
    distinguishable_state = pcvl.BasicState("|{0},{1}>")
    distinguishable_result = simulator.probs_svd(pcvl.SVDistribution(distinguishable_state))
    distinguishable_distribution = distinguishable_result["results"]
    distinguishable_probability = float(sum(distinguishable_distribution.values()))
    if not distinguishable_distribution or not isclose(
        distinguishable_probability, 1.0, abs_tol=1e-12
    ):
        raise RuntimeError("Simulator returned an invalid distinguishable-photon distribution")

    return PercevalCompatibility(
        backend_name=str(backend.name),
        backend_class=type(backend).__name__,
        full_distribution_size=len(full_distribution),
        full_distribution_probability=full_probability,
        distinguishable_distribution_size=len(distinguishable_distribution),
        distinguishable_distribution_probability=distinguishable_probability,
    )
