"""REFERENCE IMPLEMENTATION for Role 5 (Drift Analysis) -- safe to replace.

ADWIN / Page-Hinkley / DDM run on the *error stream* of a deployed model:
0 = correct prediction, 1 = mistake. In river 0.27 DDM's ``update(x)`` also
treats a truthy ``x`` as an error (it tracks "probability of error/failure"),
so all three detectors receive the same 0/1 error value. Always verify this
against the installed river version before relying on DDM in a paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from river.drift import ADWIN, PageHinkley
from river.drift.binary import DDM

METHODS = ("adwin", "ddm", "page_hinkley")


class ErrorStreamDetector:
    """Unified wrapper around river's online change detectors."""

    def __init__(self, method: str = "adwin", **params: float | int) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.method = method
        if method == "adwin":
            self._detector = ADWIN(**params)
        elif method == "page_hinkley":
            self._detector = PageHinkley(**params)
        else:
            self._detector = DDM(**params)

    def update(self, error: int | bool) -> bool:
        """Feed one prediction error and return whether drift was signalled."""
        if self.method == "ddm":
            self._detector.update(bool(error))
        else:
            self._detector.update(float(error))
        return bool(self._detector.drift_detected)


def detect_drift_indices(
    errors: Sequence[int], method: str = "adwin", **params: float | int
) -> list[int]:
    """Return the positions in ``errors`` where drift was signalled."""
    detector = ErrorStreamDetector(method, **params)
    return [i for i, error in enumerate(errors) if detector.update(error)]
