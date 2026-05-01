"""Training pipelines: supervised, RL, walk-forward, ensemble.

`walk_forward_run` is exported lazily so that importing this package
(or anything under `ai_trader.training.pipeline`) does not require the
RL stack (torch / gymnasium / stable-baselines3) to be installed.
"""
from .pipeline import build_features_pipeline, FeaturePipelineOutput

__all__ = ["build_features_pipeline", "FeaturePipelineOutput", "walk_forward_run"]


def __getattr__(name):
    if name == "walk_forward_run":
        # Imported on first access; raises ImportError here if torch/SB3 absent.
        from .walk_forward import walk_forward_run
        return walk_forward_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
