# Machine Learning Exam — Household Power Forecasting

Time-series forecasting of household electricity consumption for the 2026 ML
course project (`ml_exam.pdf`).

## Environment

Uses `uv` to manage dependencies (Python 3.10, PyTorch with CUDA).

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy matplotlib requests scikit-learn tqdm
```

Verify GPU availability:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Prepare Data

Downloads UCI household power data + Meteo-France monthly weather,
aggregates to daily, and constructs feature windows.

```bash
uv run python -m src.data --force
```

Outputs:
- `data/raw/household_power_consumption.zip` / `.txt`
- `data/raw/weather_monthly.csv` (cached from Meteo-France S3)
- `data/processed/daily_power.csv` (1433 daily rows, 18 features)

Weather is fetched once and cached; subsequent runs skip the download.

## Run Full Experiments

Runs 30 experiments (3 models × 2 horizons × 5 seeds) with early stopping.

```bash
uv run python -m src.run_experiments
```

Outputs:
- `outputs/metrics/full_metrics.csv` — per-run metrics
- `outputs/metrics/full_summary.csv` — mean/std by model+horizon
- `outputs/metrics/full_summary.json`
- `outputs/figures/*.png` — prediction vs ground truth + loss curves
- `outputs/models/*.pt` — best checkpoints

## Single Training Run

```bash
uv run python -m src.train --model lstm --horizon 90 --epochs 30 --patience 10
```

Valid models: `lstm`, `transformer`, `flow-matching`, `conv-transformer`, `spectral-patch`.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `lstm` | Model architecture |
| `--horizon` | `90` | Prediction horizon (90 or 365) |
| `--epochs` | `30` | Max training epochs |
| `--patience` | `10` | Early stopping patience |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--smoke` | — | Enable smoke mode (1 epoch, small data) |

## Generate Analysis Charts

Generates EDA figures, model comparison bar charts, prediction vs ground truth
comparisons, loss curves, and ablation plots.

```bash
uv run python -m src.plot_analysis
```

## Models

| Model | Description |
|-------|-------------|
| `lstm` | LSTM encoder + MLP head (baseline) |
| `transformer` | Transformer encoder with positional encoding (baseline) |
| `conv-transformer` | Conv1D local features + Transformer (strong baseline) |
| `spectral-patch-v2-norevin` | SOFTS core tokens + FreTS freq MLP + GeGLU + no RevIN (improved model) |
| `flow-matching` | Conditional Flow Matching trajectory forecaster (alternative) |

The primary improved model is `spectral-patch-v2-norevin`, which combines
SOFTS-style core token fusion, FreTS learnable frequency MLP, iTransformer-inspired
channel encoding, GeGLU activation, and was shown via ablation to benefit from
removing RevIN for long-horizon forecasting.

## Final Results

| Horizon | Model | MSE mean ± std | MAE mean ± std |
|--------:|-------|---------------:|---------------:|
| 90 | LSTM | 165,601 ± 2,664 | 308.79 ± 3.23 |
| 90 | Transformer | 169,783 ± 5,220 | 314.72 ± 8.63 |
| 90 | Conv-Transformer | 166,363 ± 6,417 | 310.41 ± 10.41 |
| 90 | **Spectral-Patch v2** | **177,034 ± 2,954** | **319.72 ± 3.71** |
| 90 | Flow Matching | 237,443 ± 16,308 | 379.33 ± 14.59 |
| 365 | LSTM | 163,559 ± 2,142 | 306.02 ± 2.17 |
| 365 | Transformer | 162,464 ± 3,635 | 302.37 ± 2.71 |
| 365 | Conv-Transformer | 160,845 ± 3,573 | 302.00 ± 3.27 |
| 365 | **Spectral-Patch v2** | **175,180 ± 3,738** | **319.13 ± 3.71** |
| 365 | Flow Matching | 229,221 ± 10,252 | 369.96 ± 8.51 |
| 365 | Spectral-Patch v1 | 217,779 ± 26,575 | 358.47 ± 24.83 |

## Notes

- Train/test split is chronological (80/20), not shuffled, to avoid leakage.
- Standardization uses train-only statistics.
- RevIN was found to harm long-horizon forecasting; the improved model omits it.
- Weather data from Meteo-France station 92007001 (BAGNEUX, near Sceaux).
- Weather data from Meteo-France station 92007001 (BAGNEUX, near Sceaux).
- Features include 8 power metrics, 5 weather columns, and 5 calendar features.
- Full report: `report/report.md`.

## Report

See `report/report.md` for the complete course report.

GitHub: [https://github.com/knocci/2026-Machine-Learning-Course](https://github.com/knocci/2026-Machine-Learning-Course)

## References

- UCI Household Power: https://archive.ics.uci.edu/dataset/235/
- Meteo-France: https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles
- Flow Matching: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
