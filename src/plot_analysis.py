"""Generate analysis charts for the report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .constants import (
    DAILY_POWER_CSV_PATH,
    FIGURE_DIR,
    INPUT_DAYS,
    METRIC_DIR,
    MODEL_DIR,
    TARGET_COLUMN,
    ensure_project_dirs,
)
from .data import build_windows, load_daily_matrix, prepare_daily_data, split_and_scale_windows

MODEL_DISPLAY_NAMES = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "conv-transformer": "Conv-Transformer",
    "spectral-patch-v2-norevin": "Spectral-Patch v2",
    "spectral-patch-v2": "Spectral-Patch v2 (RevIN)",
    "flow-matching": "Flow Matching",
}
MODEL_COLORS = {
    "lstm": "#1f77b4",
    "transformer": "#ff7f0e",
    "conv-transformer": "#2ca02c",
    "spectral-patch-v2-norevin": "#d62728",
    "spectral-patch-v2": "#d62728",
    "flow-matching": "#9467bd",
}
MAIN_MODELS = ["lstm", "transformer", "spectral-patch-v2-norevin"]
MODEL_ORDER = ["lstm", "transformer", "conv-transformer", "spectral-patch-v2-norevin", "flow-matching"]


def _load_order_dates() -> list[str]:
    dates = []
    with open(DAILY_POWER_CSV_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
    return dates


def _load_summary() -> list[dict]:
    with open(METRIC_DIR / "full_summary.json", "r") as f:
        return json.load(f)


def _find_checkpoint(model_name: str, horizon: int, seed: int = 11) -> Path | None:
    candidates = [
        MODEL_DIR / f"{model_name}_h{horizon}_seed{seed}.pt",
        MODEL_DIR / f"{model_name.replace('-', '_')}_h{horizon}_seed{seed}.pt",
    ]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return None


def _save_fig(fig: plt.Figure, name: str) -> Path:
    ensure_project_dirs()
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- Raw Data EDA ----

def plot_raw_timeseries() -> Path:
    dates = _load_order_dates()
    columns, matrix = load_daily_matrix()
    target_idx = columns.index(TARGET_COLUMN)
    power = matrix[:, target_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, power, linewidth=0.7, color="#1f77b4")
    ax.set_title("Daily Global Active Power (2006-12 to 2010-11)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Global Active Power (kW)")
    ticks = np.linspace(0, len(dates) - 1, 10, dtype=int)
    ax.set_xticks([dates[i] for i in ticks])
    ax.set_xticklabels([dates[i][:7] for i in ticks], rotation=45, ha="right")
    ax.grid(alpha=0.3)
    return _save_fig(fig, "eda_power_timeseries.png")


def plot_monthly_seasonal() -> Path:
    columns, matrix = load_daily_matrix()
    target_idx = columns.index(TARGET_COLUMN)
    month_idx = columns.index("month")
    power = matrix[:, target_idx]
    months = matrix[:, month_idx].astype(int)

    month_data = defaultdict(list)
    for m, p in zip(months, power):
        month_data[m].append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = list(range(1, 13))
    boxes = [month_data[m] for m in labels]
    bp = ax.boxplot(boxes, labels=labels, patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, 12))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_title("Monthly Distribution of Daily Global Active Power", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(axis="y", alpha=0.3)
    return _save_fig(fig, "eda_monthly_seasonal.png")


def plot_weekly_pattern() -> Path:
    columns, matrix = load_daily_matrix()
    target_idx = columns.index(TARGET_COLUMN)
    dow_idx = columns.index("day_of_week")
    power = matrix[:, target_idx]
    dow = matrix[:, dow_idx].astype(int)

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    means = [np.mean(power[dow == d]) for d in range(7)]
    stds = [np.std(power[dow == d]) for d in range(7)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(7), means, yerr=stds, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, 7)))
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_title("Average Daily Power by Day of Week", fontsize=13, fontweight="bold")
    ax.set_ylabel("Global Active Power (kW)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", fontsize=9)
    return _save_fig(fig, "eda_weekly_pattern.png")


def plot_rolling_stats() -> Path:
    columns, matrix = load_daily_matrix()
    target_idx = columns.index(TARGET_COLUMN)
    power = matrix[:, target_idx]

    windows = [7, 30, 90]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, w in zip(axes, windows):
        rm = np.convolve(power, np.ones(w) / w, mode="valid")
        rs = np.array([np.std(power[i - w + 1:i + 1]) for i in range(w - 1, len(power))])
        ax.plot(rm, linewidth=0.8, color="#1f77b4")
        ax.fill_between(range(len(rm)), rm - rs, rm + rs, alpha=0.15, color="#1f77b4")
        ax.set_title(f"Rolling {w}-Day Mean and Std", fontweight="bold")
        ax.set_ylabel("Power (kW)")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Day Index")
    return _save_fig(fig, "eda_rolling_stats.png")


def plot_train_test_split() -> Path:
    columns, matrix = load_daily_matrix()
    target_idx = columns.index(TARGET_COLUMN)
    power = matrix[:, target_idx]
    split = int(len(power) * 0.8)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(split), power[:split], linewidth=0.7, color="#1f77b4", label="Train Set")
    ax.plot(range(split, len(power)), power[split:], linewidth=0.7, color="#ff7f0e", label="Test Set")
    ax.axvline(x=split, color="red", linestyle="--", linewidth=1.5, label="Train/Test Split (80/20)")
    ax.set_title("Train/Test Chronological Split", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day Index")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save_fig(fig, "eda_train_test_split.png")


def plot_feature_correlations() -> Path:
    columns, matrix = load_daily_matrix()
    n = len(columns)
    corr = np.corrcoef(matrix.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(columns, fontsize=7)
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    return _save_fig(fig, "eda_correlations.png")


# ---- Metric Comparison ----

def plot_metric_bars(metric: str = "mse") -> Path:
    summary = _load_summary()
    horizons = [90, 365]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, h in zip(axes, horizons):
        h_data = [r for r in summary if r["horizon"] == h]
        h_data.sort(key=lambda r: MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99)

        models = [r["model"] for r in h_data]
        means = [r[f"{metric}_mean"] for r in h_data]
        stds = [r[f"{metric}_std"] for r in h_data]
        labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]

        bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_title(f"Horizon = {h} days", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.3)
        y_pad = max(means) * 0.12
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_pad * 0.3,
                    f"{val:.0f}", ha="center", fontsize=7)

    fig.suptitle(f"Model Comparison — {metric.upper()}", fontsize=14, fontweight="bold")
    return _save_fig(fig, f"comparison_{metric}_bar.png")


# ---- Prediction vs Ground Truth Comparison ----

def plot_comparison_prediction(horizon: int = 90, seed: int = 11) -> Path:
    from .models import build_model
    from .constants import FLOW_NUM_SAMPLES, FLOW_EULER_STEPS

    prepare_daily_data()
    columns, matrix = load_daily_matrix()
    x, y = build_windows(matrix, input_days=INPUT_DAYS, output_days=horizon)
    scaled = split_and_scale_windows(x, y)

    test_idx = 0
    test_x = scaled["test_x"][test_idx:test_idx + 1]
    test_y_raw = scaled["test_y_raw"][test_idx]
    t_mean = scaled["target_mean"]
    t_std = scaled["target_std"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_x_t = torch.from_numpy(test_x).to(device)

    predictions = {}
    for model_name in MAIN_MODELS:
        ckpt = _find_checkpoint(model_name, horizon, seed)
        if ckpt is None:
            continue
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model = build_model(model_name, input_size=test_x.shape[-1], output_days=horizon).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            if hasattr(model, "sample_forecast"):
                pred = model.sample_forecast(test_x_t, num_samples=FLOW_NUM_SAMPLES, steps=FLOW_EULER_STEPS)
            elif hasattr(model, "training_loss"):
                pred = model.sample_forecast(test_x_t, num_samples=FLOW_NUM_SAMPLES, steps=FLOW_EULER_STEPS)
            else:
                pred = model(test_x_t)
        pred_raw = pred.cpu().numpy()[0] * float(t_std) + float(t_mean)
        predictions[model_name] = pred_raw

    n_models = len(predictions)
    if n_models == 0:
        return FIGURE_DIR / "comparison_pred_none.png"

    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 4.5))
    if n_models + 1 == 1:
        axes = [axes]
    axes = list(axes)

    axes[0].plot(test_y_raw, "k-", linewidth=2)
    axes[0].set_title("Ground Truth", fontweight="bold")
    axes[0].set_xlabel("Forecast Day")
    axes[0].set_ylabel("Power (kW)")
    axes[0].grid(alpha=0.3)

    for i, (model_name, pred) in enumerate(predictions.items()):
        ax = axes[i + 1]
        ax.plot(test_y_raw, "k--", linewidth=2, label="Ground Truth")
        ax.plot(pred, "-", linewidth=2, color=MODEL_COLORS.get(model_name, "#333"), label="Prediction")
        ax.set_title(MODEL_DISPLAY_NAMES.get(model_name, model_name), fontweight="bold")
        ax.set_xlabel("Forecast Day")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Prediction vs Ground Truth Comparison — Horizon = {horizon}", fontsize=14, fontweight="bold")
    return _save_fig(fig, f"comparison_pred_h{horizon}.png")


# ---- Ablation: RevIN ----

def plot_ablation_revin() -> Path:
    categories = [
        ("Spectral-Patch v2\n(With RevIN)", 180257, 0),
        ("Spectral-Patch v2\n(Without RevIN)", 175180, 0),
        ("DLinear++\n(With RevIN)", 199991, 0),
        ("DLinear++\n(Without RevIN)", 190720, 0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, (sp_offset, pair_data) in enumerate([(0, categories[:2]), (2, categories[2:])]):
        ax = axes[ax_idx]
        labels = [p[0] for p in pair_data]
        vals = [p[1] for p in pair_data]
        colors = ["#d62728", "#2ca02c"]
        bars = ax.bar([0, 1], vals, color=colors, width=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        title = "Spectral-Patch v2" if sp_offset == 0 else "DLinear++"
        ax.set_title(f"{title} (h=365)", fontweight="bold")
        ax.set_ylabel("MSE")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                    f"{val:,}", ha="center", fontsize=11, fontweight="bold")
        delta = (vals[1] - vals[0]) / vals[0] * 100
        ax.annotate(f"\u0394 = {delta:+.1f}%", xy=(1, vals[1]),
                    xytext=(1.3, (vals[0] + vals[1]) / 2),
                    fontsize=12, color="green",
                    arrowprops=dict(arrowstyle="->", color="gray"))
    fig.suptitle("RevIN Ablation Study — Removing RevIN Improves Long-Horizon MSE", fontsize=13, fontweight="bold")
    return _save_fig(fig, "ablation_revin.png")


# ---- Combined Loss Curve (reuse existing PNGs) ----

def plot_loss_comparison() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for mi, model_name in enumerate(MAIN_MODELS):
        for hi, h in enumerate([90, 365]):
            ax = axes[hi][mi]
            loss_png = FIGURE_DIR / f"loss_{model_name}_h{h}_seed11.png"
            if loss_png.exists():
                img = plt.imread(str(loss_png))
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=14)
            display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            ax.set_title(f"{display}  (h={h})", fontweight="bold", fontsize=11)
    fig.suptitle("Training Loss Curves (Representative Seed)", fontsize=14, fontweight="bold")
    return _save_fig(fig, "loss_curves_comparison.png")


# ---- Main ----

def generate_all() -> dict[str, str]:
    ensure_project_dirs()
    results = {}

    print("Generating EDA charts...")
    for func in [plot_raw_timeseries, plot_monthly_seasonal, plot_weekly_pattern,
                 plot_rolling_stats, plot_train_test_split, plot_feature_correlations]:
        try:
            name = func.__name__.replace("plot_", "eda_")
            results[name] = str(func())
            print(f"  {results[name]}")
        except Exception as e:
            print(f"  ERROR in {func.__name__}: {e}")

    print("Generating comparison charts...")
    try:
        results["comparison_mse_bar"] = str(plot_metric_bars("mse"))
        print(f"  {results['comparison_mse_bar']}")
    except Exception as e:
        print(f"  ERROR mse bar: {e}")

    try:
        results["comparison_mae_bar"] = str(plot_metric_bars("mae"))
        print(f"  {results['comparison_mae_bar']}")
    except Exception as e:
        print(f"  ERROR mae bar: {e}")

    for h in [90, 365]:
        try:
            results[f"comparison_pred_h{h}"] = str(plot_comparison_prediction(h))
            print(f"  {results[f'comparison_pred_h{h}']}")
        except Exception as e:
            print(f"  ERROR comparison pred h={h}: {e}")

    try:
        results["loss_curves_comparison"] = str(plot_loss_comparison())
        print(f"  {results['loss_curves_comparison']}")
    except Exception as e:
        print(f"  ERROR loss curves: {e}")

    try:
        results["ablation_revin"] = str(plot_ablation_revin())
        print(f"  {results['ablation_revin']}")
    except Exception as e:
        print(f"  ERROR ablation: {e}")

    return results


def main() -> None:
    output = generate_all()
    print(f"\nGenerated {len(output)} charts in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
