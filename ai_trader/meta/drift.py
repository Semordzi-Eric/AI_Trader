"""Concept-drift detection.

Two complementary signals:

1. Page-Hinkley test on the running mean of an online metric (e.g. the policy's
   step PnL or supervised model accuracy). Detects sudden regime shifts
   asymmetrically (we only care about *negative* drift in PnL).

2. Two-sample Kolmogorov-Smirnov test comparing a recent feature window to a
   reference window. Detects distributional shifts even when the policy still
   appears to perform — the ground beneath it is moving.

When either fires, the higher-level training loop should retrain or fall back
to a safe baseline (e.g. flat) until things stabilize.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class PageHinkley:
    """Online change-point detector for a 1-D signal (e.g. step PnL).

    Tracks the cumulative deviation from the running mean. When the deviation
    drops by more than `threshold` from its maximum, we flag drift. `delta` is a
    small positive constant that prevents false positives from noise.
    """

    delta: float = 0.005
    threshold: float = 50.0
    _mean: float = 0.0
    _n: int = 0
    _cum: float = 0.0
    _max: float = 0.0

    def update(self, x: float) -> bool:
        """Push value, return True if drift detected. Auto-resets on drift."""
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._cum += x - self._mean - self.delta
        self._max = max(self._max, self._cum)

        drift = (self._max - self._cum) > self.threshold
        if drift:
            self.reset()
        return drift

    def reset(self) -> None:
        self._mean = 0.0
        self._n = 0
        self._cum = 0.0
        self._max = 0.0


@dataclass
class FeatureKSTest:
    """KS test between a fixed reference window and a sliding recent window."""

    p_threshold: float = 0.01
    reference: Optional[np.ndarray] = None      # (N_ref, F)

    def fit_reference(self, X: np.ndarray) -> "FeatureKSTest":
        self.reference = np.asarray(X, dtype=np.float64).copy()
        return self

    def detect(self, recent: np.ndarray) -> tuple[bool, dict]:
        if self.reference is None:
            raise RuntimeError("Call fit_reference first")
        if recent.shape[1] != self.reference.shape[1]:
            raise ValueError("feature dim mismatch")
        flags = []
        details = {}
        for i in range(recent.shape[1]):
            ref = self.reference[:, i]
            rec = recent[:, i]
            ref = ref[~np.isnan(ref)]
            rec = rec[~np.isnan(rec)]
            if len(ref) < 30 or len(rec) < 30:
                continue
            stat, p = stats.ks_2samp(ref, rec)
            details[f"f{i}"] = {"stat": float(stat), "p": float(p)}
            if p < self.p_threshold:
                flags.append(i)
        return (len(flags) > 0, {"flagged_features": flags, "details": details})


@dataclass
class DriftDetector:
    """Combines Page-Hinkley on a performance signal with KS on features."""

    ph_delta: float = 0.005
    ph_threshold: float = 50.0
    ks_p: float = 0.01
    _ph: PageHinkley = field(init=False)
    _ks: FeatureKSTest = field(init=False)

    def __post_init__(self) -> None:
        self._ph = PageHinkley(delta=self.ph_delta, threshold=self.ph_threshold)
        self._ks = FeatureKSTest(p_threshold=self.ks_p)

    def fit_reference(self, ref_features: np.ndarray) -> "DriftDetector":
        self._ks.fit_reference(ref_features)
        return self

    def update(self, perf_signal: float, recent_features: Optional[np.ndarray] = None) -> dict:
        result = {"performance_drift": False, "feature_drift": False, "ks_details": {}}
        result["performance_drift"] = self._ph.update(perf_signal)
        if recent_features is not None and self._ks.reference is not None:
            flagged, details = self._ks.detect(recent_features)
            result["feature_drift"] = flagged
            result["ks_details"] = details
        return result
