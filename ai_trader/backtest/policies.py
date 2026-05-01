"""Policy interfaces and reference implementations.

The backtester is policy-agnostic. A `Policy` is anything with `.act(obs) -> float`
in [-1, 1]. We provide a no-op baseline (always flat), a random policy, and a wrapper
around a trained PPO agent. This lets us answer 'is the model better than zero?'
and 'is it better than coin flips?' before any deeper analysis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np

# Avoid pulling in torch / gymnasium / stable-baselines3 just to import this
# module. The RL stack is only needed when RLPolicy is actually instantiated.
if TYPE_CHECKING:
    from ..rl.agent import PPOAgent


class Policy(ABC):
    """Map an observation to a target position fraction in [-1, 1]."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> float: ...


class BaselinePolicy(Policy):
    """Always flat. Sets the floor: any strategy must beat zero net of costs."""

    def act(self, obs: np.ndarray) -> float:  # noqa: ARG002
        return 0.0


class RandomPolicy(Policy):
    """Uniform random in [-1, 1]. Expected to lose to costs."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> float:  # noqa: ARG002
        return float(self.rng.uniform(-1.0, 1.0))


class RLPolicy(Policy):
    """Wrap a trained PPO agent."""

    def __init__(self, agent: "PPOAgent", deterministic: bool = True) -> None:
        self.agent = agent
        self.deterministic = deterministic

    def act(self, obs: np.ndarray) -> float:
        action = self.agent.predict(obs, deterministic=self.deterministic)
        return float(np.clip(action[0], -1.0, 1.0))


class TrendFollowPolicy(Policy):
    """Reads `trend_strength` from the obs as a sanity-check baseline.

    Position scales with sign and magnitude of recent trend strength.
    Useful as a non-trivial benchmark.

    The flat observation is laid out as [window_size × n_features ... portfolio_3],
    so the *latest* bar's value of feature `feature_idx` lives at index
    `(window_size - 1) * n_features + feature_idx`. We compute that on first call
    given the obs shape and the supplied feature_idx + n_features.
    """

    def __init__(self, trend_idx: int, n_features: int | None = None,
                 window_size: int | None = None, threshold: float = 0.1) -> None:
        self.feature_idx = int(trend_idx)
        self.n_features = n_features
        self.window_size = window_size
        self.threshold = threshold

    def _resolved_idx(self, obs: np.ndarray) -> int:
        # If caller didn't supply n_features/window_size, infer assuming the last
        # 3 obs entries are portfolio state (matches TradingEnv & make_observation_matrix).
        if self.n_features is not None and self.window_size is not None:
            return (self.window_size - 1) * self.n_features + self.feature_idx
        # Fall back: assume obs == window*F + 3 and we want the last slot of feature_idx.
        # We can't know F without help, so trust feature_idx is already absolute.
        return self.feature_idx

    def act(self, obs: np.ndarray) -> float:
        idx = self._resolved_idx(obs)
        v = float(obs[idx])
        if abs(v) < self.threshold:
            return 0.0
        return float(np.clip(v, -1.0, 1.0))
