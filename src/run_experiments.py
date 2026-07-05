"""Run full experiments across models x horizons x seeds."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    FIGURE_DIR,
    FULL_BATCH_SIZE,
    FULL_EPOCHS,
    FULL_SEEDS,
    INPUT_DAYS,
    LONG_OUTPUT_DAYS,
    METRIC_DIR,
    SMOKE_OUTPUT_DAYS,
    ensure_project_dirs,
)
from .data import build_windows, load_daily_matrix, prepare_daily_data, split_and_scale_windows
from .train import (
    TrainConfig,
    train_once,
    write_metrics_csv,
    resolve_device,
    set_seed,
)


EXPERIMENT_MODELS = ("lstm", "transformer", "conv-transformer", "spectral-patch-v2-norevin", "flow-matching")
EXPERIMENT_HORIZONS = (SMOKE_OUTPUT_DAYS, LONG_OUTPUT_DAYS)
EXPERIMENT_SEEDS = FULL_SEEDS


def _compute_summary(metrics: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in metrics:
        key = (row["model"], row["horizon"])
        groups[key].append(row)

    summary = []
    for (model, horizon), rows in sorted(groups.items()):
        mses = [r["mse"] for r in rows]
        maes = [r["mae"] for r in rows]
        summary.append({
            "model": model,
            "horizon": horizon,
            "num_runs": len(rows),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        })
    return summary


def _plot_combined(
    model_name: str,
    horizon: int,
    predictions: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
) -> str:
    path = FIGURE_DIR / f"combined_{model_name}_h{horizon}.png"
    fig, axes = plt.subplots(
        min(3, len(predictions)), 1,
        figsize=(12, 3 * min(3, len(predictions))),
        squeeze=False,
    )
    for i, (pred, target) in enumerate(predictions[:3]):
        ax = axes[i][0]
        ax.plot(target, label="Ground Truth", linewidth=2)
        ax.plot(pred, label=f"Prediction (seed={seeds[i]})", linewidth=2)
        ax.set_title(f"{model_name} h={horizon} seed={seeds[i]}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def main() -> None:
    ensure_project_dirs()
    print("=" * 60)
    print("Full Experiment Runner")
    print(f"Models: {EXPERIMENT_MODELS}")
    print(f"Horizons: {EXPERIMENT_HORIZONS}")
    print(f"Seeds: {EXPERIMENT_SEEDS}")
    print(f"Total runs: {len(EXPERIMENT_MODELS) * len(EXPERIMENT_HORIZONS) * len(EXPERIMENT_SEEDS)}")
    print("=" * 60)

    all_metrics = []
    total = len(EXPERIMENT_MODELS) * len(EXPERIMENT_HORIZONS) * len(EXPERIMENT_SEEDS)
    run_idx = 0
    start_time = time.time()

    for model_name in EXPERIMENT_MODELS:
        for horizon in EXPERIMENT_HORIZONS:
            run_predictions: list[tuple[np.ndarray, np.ndarray]] = []
            run_seeds: list[int] = []
            for seed in EXPERIMENT_SEEDS:
                run_idx += 1
                print(f"\n--- Run {run_idx}/{total}: {model_name} h={horizon} seed={seed} ---")

                if "v2" in model_name or "dlinear" in model_name or "v3" in model_name:
                    lr = 5e-4
                    patience = 15
                    epochs = 50
                else:
                    lr = 1e-3
                    patience = 10
                    epochs = FULL_EPOCHS

                use_sam = "dlinear" in model_name
                use_mixup = "dlinear" in model_name

                config = TrainConfig(
                    model=model_name,
                    horizon=horizon,
                    epochs=epochs,
                    batch_size=FULL_BATCH_SIZE,
                    patience=patience,
                    val_split=0.1,
                    lr=lr,
                    seed=seed,
                    device="auto",
                    smoke=False,
                    max_train_windows=None,
                    use_sam=use_sam,
                    use_mixup=use_mixup,
                    sam_rho=0.05,
                    mixup_alpha=0.2,
                )
                result = train_once(config)
                all_metrics.append(result)
                try:
                    pred_path = FIGURE_DIR / f"{model_name}_h{horizon}_seed{seed}.png"
                    if pred_path.exists():
                        run_predictions.append((None, None))
                        run_seeds.append(seed)
                except Exception:
                    pass

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    metrics_path = write_metrics_csv(all_metrics, METRIC_DIR / "full_metrics.csv")
    print(f"Wrote full metrics: {metrics_path}")

    summary = _compute_summary(all_metrics)
    summary_csv = METRIC_DIR / "full_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "horizon", "num_runs", "mse_mean", "mse_std", "mae_mean", "mae_std"],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote summary: {summary_csv}")

    summary_json = METRIC_DIR / "full_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary JSON: {summary_json}")

    print("\n=== Final Summary ===")
    for row in summary:
        print(
            f"{row['model']:>16s}  h={row['horizon']:>3d}  "
            f"MSE={row['mse_mean']:.1f} +/- {row['mse_std']:.1f}  "
            f"MAE={row['mae_mean']:.2f} +/- {row['mae_std']:.2f}"
        )


if __name__ == "__main__":
    main()
