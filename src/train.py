"""Training and evaluation entry points."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .constants import (
    DEFAULT_DEVICE,
    DEFAULT_SEED,
    FIGURE_DIR,
    FLOW_EULER_STEPS,
    FLOW_NUM_SAMPLES,
    FULL_BATCH_SIZE,
    FULL_EPOCHS,
    INPUT_DAYS,
    METRIC_DIR,
    MODEL_DIR,
    SMOKE_BATCH_SIZE,
    SMOKE_EPOCHS,
    SMOKE_MAX_WINDOWS,
    SMOKE_OUTPUT_DAYS,
    ensure_project_dirs,
)
from .data import build_windows, load_daily_matrix, prepare_daily_data, split_and_scale_windows
from .models import build_model


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper (Foret et al., ICLR 2021).

    Wraps any base optimizer and performs the two-step SAM update:
    1. Compute gradient at current position, perturb weights
    2. Compute gradient at perturbed position, restore weights, apply SAM gradient
    """

    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        defaults = {"rho": rho}
        super().__init__(base_optimizer.param_groups, defaults)
        self.base_optimizer = base_optimizer

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        scale = self.defaults["rho"] / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                self.state[p]["e_w"] = e_w
                p.data = p.data + e_w
        if zero_grad:
            self.zero_grad()

    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = p.data - self.state[p]["e_w"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups
                for p in group["params"] if p.grad is not None
            ]),
        )
        return norm

    def step(self, closure=None):
        raise RuntimeError("SAM requires calling first_step() and second_step()")


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup (Zhang et al., ICLR 2018) to a batch."""
    if alpha <= 0:
        return x, y
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


@dataclass
class TrainConfig:
    model: str = "lstm"
    horizon: int = SMOKE_OUTPUT_DAYS
    epochs: int = SMOKE_EPOCHS
    batch_size: int = SMOKE_BATCH_SIZE
    lr: float = 1e-3
    seed: int = DEFAULT_SEED
    device: str = DEFAULT_DEVICE
    smoke: bool = True
    max_train_windows: int | None = SMOKE_MAX_WINDOWS
    patience: int = 0
    val_split: float = 0.1
    use_sam: bool = False
    use_mixup: bool = False
    sam_rho: float = 0.05
    mixup_alpha: float = 0.2

    @property
    def is_full(self) -> bool:
        return not self.smoke


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str = DEFAULT_DEVICE) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return requested


def _make_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader | None, DataLoader, dict[str, np.ndarray]]:
    prepare_daily_data()
    _, matrix = load_daily_matrix()
    x, y = build_windows(matrix, input_days=INPUT_DAYS, output_days=config.horizon)
    scaled = split_and_scale_windows(
        x,
        y,
        max_train_windows=config.max_train_windows if config.smoke else None,
    )

    train_x_t = torch.from_numpy(scaled["train_x"])
    train_y_t = torch.from_numpy(scaled["train_y"])
    test_dataset = TensorDataset(
        torch.from_numpy(scaled["test_x"]),
        torch.from_numpy(scaled["test_y"]),
    )

    val_loader = None
    if config.val_split > 0 and len(train_x_t) > 4:
        val_size = max(1, int(len(train_x_t) * config.val_split))
        indices = list(range(len(train_x_t)))
        split_point = len(indices) - val_size
        train_dataset = TensorDataset(train_x_t[:split_point], train_y_t[:split_point])
        val_dataset = TensorDataset(train_x_t[split_point:], train_y_t[split_point:])
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        train_dataset = TensorDataset(train_x_t, train_y_t)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scaled


def _evaluate_scaled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            if hasattr(model, "sample_forecast"):
                output_tensor = model.sample_forecast(
                    batch_x,
                    num_samples=FLOW_NUM_SAMPLES,
                    steps=FLOW_EULER_STEPS,
                )
            else:
                output_tensor = model(batch_x)
            output = output_tensor.cpu().numpy()
            predictions.append(output)
            targets.append(batch_y.numpy())

    pred_scaled = np.concatenate(predictions, axis=0)
    target_scaled = np.concatenate(targets, axis=0)
    pred_raw = pred_scaled * float(target_std) + float(target_mean)
    target_raw = target_scaled * float(target_std) + float(target_mean)
    mse = float(np.mean((pred_raw - target_raw) ** 2))
    mae = float(np.mean(np.abs(pred_raw - target_raw)))
    return mse, mae, pred_raw, target_raw


def _val_loss_scaled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_huber: bool = False,
) -> float:
    model.eval()
    criterion = nn.HuberLoss(delta=1.0) if use_huber else nn.MSELoss()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if hasattr(model, "training_loss"):
                loss = model.training_loss(batch_x, batch_y)
            else:
                output = model(batch_x)
                loss = criterion(output, batch_y)
            total += float(loss.detach().cpu()) * batch_x.size(0)
            count += batch_x.size(0)
    return total / max(count, 1)


def _plot_prediction(
    model_name: str,
    horizon: int,
    seed: int,
    prediction: np.ndarray,
    target: np.ndarray,
) -> Path:
    ensure_project_dirs()
    path = FIGURE_DIR / f"{model_name}_h{horizon}_seed{seed}.png"
    plt.figure(figsize=(10, 4))
    plt.plot(target, label="Ground Truth", linewidth=2)
    plt.plot(prediction, label="Prediction", linewidth=2)
    plt.title(f"{model_name} forecast, horizon={horizon}")
    plt.xlabel("Forecast day")
    plt.ylabel("Daily global active power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_loss_curve(
    model_name: str,
    horizon: int,
    seed: int,
    train_losses: list[float],
    val_losses: list[float] | None,
) -> Path:
    ensure_project_dirs()
    path = FIGURE_DIR / f"loss_{model_name}_h{horizon}_seed{seed}.png"
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=1.5)
    if val_losses:
        plt.plot(val_losses, label="Val Loss", linewidth=1.5)
    plt.title(f"{model_name} loss curve, horizon={horizon}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train_once(config: TrainConfig) -> dict[str, str | int | float]:
    ensure_project_dirs()
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader, test_loader, scaled = _make_loaders(config)

    sample_x, _ = next(iter(train_loader))
    model = build_model(
        config.model, input_size=sample_x.shape[-1], output_days=config.horizon
    ).to(device)
    criterion = nn.MSELoss()
    use_huber = "v2" in config.model or "dlinear" in config.model or "v3" in config.model
    huber_criterion = nn.HuberLoss(delta=1.0) if use_huber else None
    base_opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    optimizer: SAM | torch.optim.Optimizer = (
        SAM(base_opt, rho=config.sam_rho) if config.use_sam else base_opt
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt if config.use_sam else optimizer, T_max=config.epochs
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    effective_patience = config.patience if config.patience > 0 else config.epochs

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if config.use_mixup:
                batch_x, batch_y = mixup_data(batch_x, batch_y, alpha=config.mixup_alpha)

            def _forward_loss() -> torch.Tensor:
                if hasattr(model, "training_loss"):
                    return model.training_loss(batch_x, batch_y)
                output_t = model(batch_x)
                return (
                    huber_criterion(output_t, batch_y)
                    if huber_criterion
                    else criterion(output_t, batch_y)
                )

            if config.use_sam:
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_loss()
                loss.backward()
                optimizer.first_step(zero_grad=True)
                _forward_loss().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        avg_train = float(np.mean(batch_losses))
        train_losses.append(avg_train)

        if val_loader is not None:
            avg_val = _val_loss_scaled(model, val_loader, device, use_huber=use_huber)
            val_losses.append(avg_val)
            monitor = avg_val
        else:
            monitor = avg_train

        if monitor < best_val_loss:
            best_val_loss = monitor
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        val_str = f"val_loss={avg_val:.6f}" if val_loader else ""
        print(
            f"{config.model} h={config.horizon} s={config.seed} "
            f"epoch {epoch + 1}/{config.epochs} "
            f"train_loss={avg_train:.6f} {val_str} device={device}"
        )

        if patience_counter >= effective_patience:
            print(f"Early stopping at epoch {epoch + 1} (best: epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    mse, mae, pred_raw, target_raw = _evaluate_scaled(
        model,
        test_loader,
        device,
        scaled["target_mean"],
        scaled["target_std"],
    )

    checkpoint_path = MODEL_DIR / f"{config.model}_h{config.horizon}_seed{config.seed}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {k: v for k, v in config.__dict__.items()},
            "target_mean": scaled["target_mean"].tolist(),
            "target_std": scaled["target_std"].tolist(),
        },
        checkpoint_path,
    )
    figure_path = _plot_prediction(
        config.model, config.horizon, config.seed, pred_raw[0], target_raw[0]
    )
    loss_path = _plot_loss_curve(
        config.model, config.horizon, config.seed, train_losses, val_losses
    )

    result: dict[str, str | int | float] = {
        "model": config.model,
        "horizon": config.horizon,
        "seed": config.seed,
        "epochs": len(train_losses),
        "best_epoch": best_epoch,
        "device": str(device),
        "mse": mse,
        "mae": mae,
        "checkpoint": str(checkpoint_path),
        "figure": str(figure_path),
        "loss_curve": str(loss_path),
    }
    print(json.dumps(result, indent=2))
    return result


def write_metrics_csv(results: list[dict[str, str | int | float]], path: Path) -> Path:
    ensure_project_dirs()
    fieldnames = [
        "model", "horizon", "seed", "epochs", "best_epoch",
        "device", "mse", "mae", "checkpoint", "figure", "loss_curve",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train one household power forecaster.")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer", "conv-transformer", "spectral-patch", "spectral-patch-v2", "dlinear-plus", "flow-matching"],
        default="lstm",
    )
    parser.add_argument("--horizon", type=int, default=SMOKE_OUTPUT_DAYS)
    parser.add_argument("--epochs", type=int, default=SMOKE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=SMOKE_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=SMOKE_MAX_WINDOWS)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.0)
    args = parser.parse_args()
    return TrainConfig(
        model=args.model,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        smoke=args.smoke,
        max_train_windows=args.max_train_windows,
        patience=args.patience,
        val_split=args.val_split,
    )


def main() -> None:
    config = parse_args()
    result = train_once(config)
    metrics_path = METRIC_DIR / f"{config.model}_h{config.horizon}_seed{config.seed}.csv"
    write_metrics_csv([result], metrics_path)
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
