"""Runtime compatibility checks for the pinned Perceval release."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import perceval as pcvl

_ABSOLUTE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PercevalCompatibility:
    """Observed results from the Perceval compatibility probe."""

    backend_name: str
    backend_class: str
    full_distribution_size: int
    full_distribution_probability: float
    indistinguishable_left_bunch_probability: float
    indistinguishable_coincidence_probability: float
    indistinguishable_right_bunch_probability: float
    distinguishable_distribution_size: int
    distinguishable_distribution_probability: float
    distinguishable_left_bunch_probability: float
    distinguishable_coincidence_probability: float
    distinguishable_right_bunch_probability: float


def _require_probability(actual: float, expected: float, label: str) -> None:
    if not isclose(actual, expected, rel_tol=0.0, abs_tol=_ABSOLUTE_TOLERANCE):
        raise RuntimeError(f"{label}: expected {expected}, got {actual}")


def verify_perceval_compatibility(seed: int = 0) -> PercevalCompatibility:
    """Exercise the Perceval 1.3 APIs required by later pilot milestones."""
    pcvl.random_seed(seed)

    backend = pcvl.BackendFactory.get_backend("SLOS")
    simulator = pcvl.Simulator(backend)
    simulator.set_circuit(pcvl.BS())

    left_bunch = pcvl.BasicState([2, 0])
    coincidence = pcvl.BasicState([1, 1])
    right_bunch = pcvl.BasicState([0, 2])

    full_distribution = simulator.probs(coincidence)
    full_probability = float(sum(full_distribution.values()))
    if not full_distribution:
        raise RuntimeError("SLOS returned an invalid full probability distribution")
    _require_probability(full_probability, 1.0, "indistinguishable distribution normalization")

    indistinguishable_left_bunch = float(full_distribution.get(left_bunch, 0.0))
    indistinguishable_coincidence = float(full_distribution.get(coincidence, 0.0))
    indistinguishable_right_bunch = float(full_distribution.get(right_bunch, 0.0))
    _require_probability(indistinguishable_left_bunch, 0.5, "left HOM bunching")
    _require_probability(indistinguishable_coincidence, 0.0, "HOM coincidence")
    _require_probability(indistinguishable_right_bunch, 0.5, "right HOM bunching")

    # Perceval 1.3 represents distinguishability tags with NoisyFockState syntax.
    distinguishable_state = pcvl.BasicState("|{0},{1}>")
    distinguishable_result = simulator.probs_svd(pcvl.SVDistribution(distinguishable_state))
    distinguishable_distribution = distinguishable_result["results"]
    distinguishable_probability = float(sum(distinguishable_distribution.values()))
    if not distinguishable_distribution:
        raise RuntimeError("Simulator returned an invalid distinguishable-photon distribution")
    _require_probability(
        distinguishable_probability, 1.0, "distinguishable distribution normalization"
    )

    distinguishable_left_bunch = float(distinguishable_distribution.get(left_bunch, 0.0))
    distinguishable_coincidence = float(distinguishable_distribution.get(coincidence, 0.0))
    distinguishable_right_bunch = float(distinguishable_distribution.get(right_bunch, 0.0))
    _require_probability(distinguishable_left_bunch, 0.25, "left distinguishable bunching")
    _require_probability(distinguishable_coincidence, 0.5, "distinguishable coincidence")
    _require_probability(distinguishable_right_bunch, 0.25, "right distinguishable bunching")

    return PercevalCompatibility(
        backend_name=str(backend.name),
        backend_class=type(backend).__name__,
        full_distribution_size=len(full_distribution),
        full_distribution_probability=full_probability,
        indistinguishable_left_bunch_probability=indistinguishable_left_bunch,
        indistinguishable_coincidence_probability=indistinguishable_coincidence,
        indistinguishable_right_bunch_probability=indistinguishable_right_bunch,
        distinguishable_distribution_size=len(distinguishable_distribution),
        distinguishable_distribution_probability=distinguishable_probability,
        distinguishable_left_bunch_probability=distinguishable_left_bunch,
        distinguishable_coincidence_probability=distinguishable_coincidence,
        distinguishable_right_bunch_probability=distinguishable_right_bunch,
    )
