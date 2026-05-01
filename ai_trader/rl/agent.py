"""PPO agent built on Stable-Baselines3, with vectorized environments and a
predict() helper that matches the rest of the codebase.

We use SB3 because writing a correct PPO implementation from scratch is a
multi-week trap with subtle bugs that silently degrade performance. The
abstraction here is thin enough to swap for SAC, DQN, or RecurrentPPO later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv

from ..utils.logging_setup import get_logger
from .env import TradingEnv

logger = get_logger(__name__)


@dataclass
class PPOAgent:
    """Thin wrapper providing train/save/load/predict."""

    env_factory: Callable[[], TradingEnv]
    n_envs: int = 8
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    policy: str = "MlpPolicy"
    seed: int = 42
    use_subproc: bool = False        # set True for CPU-bound speedup with many envs
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [128, 128]})

    def __post_init__(self) -> None:
        self.model: Optional[PPO] = None

    def _build_vec_env(self):
        env_fns = [self.env_factory for _ in range(self.n_envs)]
        if self.use_subproc and self.n_envs > 1:
            vec = SubprocVecEnv(env_fns, start_method="forkserver")
        else:
            vec = DummyVecEnv(env_fns)
        return VecMonitor(vec)

    def train(self, save_path: str | Path | None = None) -> "PPOAgent":
        vec = self._build_vec_env()
        self.model = PPO(
            self.policy,
            vec,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            ent_coef=self.ent_coef,
            clip_range=self.clip_range,
            verbose=1,
            seed=self.seed,
            policy_kwargs=self.policy_kwargs,
        )
        logger.info("Training PPO for %d timesteps over %d envs", self.total_timesteps, self.n_envs)
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=False)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_path))
            logger.info("Saved PPO model to %s", save_path)

        vec.close()
        return self

    def load(self, path: str | Path) -> "PPOAgent":
        self.model = PPO.load(str(path))
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Agent has no model. Call train() or load().")
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action
