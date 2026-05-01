"""Supervised training loop with early stopping and best-model checkpointing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SupervisedTrainer:
    """Trainer for sign-of-return classification or return regression."""

    model: nn.Module
    target_kind: str = "sign_return"        # sign_return | return | volatility
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 20
    early_stopping_patience: int = 5
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.target_kind == "sign_return":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[str | Path] = None,
    ) -> dict:
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        best_val = float("inf")
        best_state: Optional[dict] = None
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            # ---- train ----
            self.model.train()
            train_losses = []
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if self.target_kind == "sign_return":
                    y = (y > 0).float()
                opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                train_losses.append(loss.item())
            scheduler.step()

            # ---- val ----
            val_loss, val_acc = self._evaluate(val_loader)
            train_loss = float(np.mean(train_losses))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                "epoch %3d | train %.5f | val %.5f | acc %.3f",
                epoch + 1, train_loss, val_loss, val_acc,
            )

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)
                logger.info("Saved best model to %s", save_path)

        return {"history": history, "best_val_loss": best_val}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            if self.target_kind == "sign_return":
                y_bin = (y > 0).float()
                pred = self.model(x)
                losses.append(self.criterion(pred, y_bin).item())
                correct += ((pred > 0).float() == y_bin).sum().item()
                total += y.size(0)
            else:
                pred = self.model(x)
                losses.append(self.criterion(pred, y).item())
                total += y.size(0)
        acc = correct / total if total > 0 and self.target_kind == "sign_return" else float("nan")
        return float(np.mean(losses)), acc

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        outs = []
        for x, _ in loader:
            x = x.to(self.device)
            outs.append(self.model(x).cpu().numpy())
        return np.concatenate(outs)
