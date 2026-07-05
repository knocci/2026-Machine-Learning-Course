# GPU Continuation Guide For The ML Exam Project

This file is the handoff guide for the next agent working in a GPU environment.
The goal is to finish the full machine learning course assignment described in
`ml_exam.pdf`, not merely the current CPU smoke test.

## Current State

- Workspace: `D:\desktop\course_note\machinelearning\exam`
- Existing code:
  - `src/constants.py`: central paths, URLs, horizons, seeds, batch sizes.
  - `src/data.py`: UCI download/extraction, daily aggregation, window building.
  - `src/models.py`: LSTM, Transformer, Conv-Transformer.
  - `src/train.py`: single-model training and evaluation.
  - `src/run_smoke.py`: CPU smoke test across all three models.
- Existing local data:
  - `data/raw/household_power_consumption.zip`
  - `data/raw/household_power_consumption.txt`
  - `data/processed/daily_power.csv`
- Existing smoke outputs:
  - `outputs/metrics/smoke_metrics.csv`
  - `outputs/figures/*_h90_seed42.png`
  - `outputs/models/*_h90_seed42.pt`
- Smoke test was completed on CPU with `conda run -n yolov8 python -m src.run_smoke`.
  Treat those metrics only as sanity checks, not as reportable final results.

## Assignment Requirements

Finish the full project from `ml_exam.pdf`:

- Predict future daily `global_active_power` from the previous 90 days.
- Train separate models for:
  - short horizon: predict next 90 days;
  - long horizon: predict next 365 days.
- Required methods:
  - LSTM;
  - Transformer;
  - improved self-designed model.
- Run at least 5 experiments per method and horizon.
- Report MSE and MAE as mean and standard deviation.
- Plot prediction vs Ground Truth curves.
- Submit code and a report with:
  - problem introduction;
  - model descriptions;
  - results and analysis;
  - discussion.

## Data Sources

Use these sources first:

- UCI Individual Household Electric Power Consumption:
  - page: `https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption`
  - zip fallback already in code:
    `https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip`
  - UCI describes the data as 2,075,259 minute-level measurements from a house in Sceaux, near Paris, between December 2006 and November 2010, with missing values.
- Meteo-France monthly climate data:
  - page: `https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-mensuelles`
  - the page describes monthly climate data by station, distributed as compressed CSV resources and updated regularly.

Current code already downloads and prepares the UCI data. Weather columns are
currently optional and default to zero when `data/raw/weather_monthly.csv` is
missing. For the final assignment, implement real weather ingestion if possible.

## GPU Environment

Use the GPU machine's existing conda setup if the user provides one. If no
suitable environment exists, create one explicitly for this project:

```powershell
conda create -n ml-exam-gpu python=3.10 -y
conda activate ml-exam-gpu
```

Install PyTorch for the GPU machine's CUDA version. Check the driver first:

```powershell
nvidia-smi
```

For CUDA 12.1, for example:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy matplotlib requests pandas scikit-learn tqdm
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Do not hardcode local paths. Put path changes in `src/constants.py`.

## Implementation Steps

1. Verify the existing code still runs:

```powershell
python -m src.data
python -m src.run_smoke
```

2. Add full experiment orchestration:

- Create a script such as `src/run_experiments.py`.
- Loop over models: `lstm`, `transformer`, `conv-transformer`.
- Loop over horizons: `90`, `365`.
- Loop over seeds from `FULL_SEEDS` in `src/constants.py`.
- Save one row per run to `outputs/metrics/full_metrics.csv`.
- Save aggregate mean/std by model and horizon to
  `outputs/metrics/full_summary.csv` and `outputs/metrics/full_summary.json`.

3. Extend training for full runs:

- Use GPU automatically through `DEFAULT_DEVICE = "auto"`.
- Keep train/test split chronological; do not shuffle time order before split.
- Allow CLI flags for `--model`, `--horizon`, `--epochs`, `--batch-size`,
  `--seed`, `--lr`, and `--device`.
- Add early stopping on validation MSE if training is slow or unstable.
- Save best checkpoint, not just the last checkpoint.
- Save loss curves for each run.

4. Improve data support:

- Keep UCI daily aggregation exactly aligned with the PDF:
  - sum: `global_active_power`, `global_reactive_power`,
    `sub_metering_1`, `sub_metering_2`, `sub_metering_3`;
  - mean: `voltage`, `global_intensity`;
  - compute `sub_metering_remainder`.
- Add a real weather downloader:
  - query the data.gouv.fr dataset/API for monthly climate resources;
  - prefer station(s) near Sceaux/Paris, especially department 92 or 75;
  - extract monthly `RR`, `NBJRR1`, `NBJRR5`, `NBJRR10`, `NBJBROU`;
  - join weather by `YYYY-MM` to each daily sample.
- If weather download fails, keep the zero-filled fallback, but document the
  failure in the report discussion.

5. Add report generation:

- Create `report/report.md`.
- Include final tables from `outputs/metrics/full_summary.csv`.
- Include selected plots from `outputs/figures`.
- Export to PDF if a local Markdown-to-PDF tool is available; otherwise keep the
  Markdown ready for manual conversion.
- Include a clear tool-use disclosure: ChatGPT/Codex was used for code/report
  assistance.

## Baselines And Evaluation

Before deep tuning, add simple baselines for context:

- Persistence baseline: repeat the last observed daily power value.
- Seasonal baseline: repeat the most recent 90-day pattern where possible.

Use these baselines only for analysis unless the assignment only wants the three
neural models in the main comparison table.

Metrics:

- Compute MSE and MAE on the original daily `global_active_power` scale.
- Report mean and std across five seeds.
- Plot at least one representative test sample per model and horizon.
- Also generate a combined comparison plot if it improves report clarity.

## If Results Are Poor

Try improvements in this order, recording each change:

1. Data and target scaling:
   - verify no leakage from test data into scalers;
   - standardize features with train-only statistics;
   - standardize the target and invert before metrics;
   - consider predicting normalized deltas from the last input day.

2. Training stability:
   - lower learning rate to `5e-4` or `1e-4`;
   - add gradient clipping, e.g. `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`;
   - increase epochs with early stopping;
   - tune batch size for GPU memory.

3. Model capacity:
   - LSTM: try hidden sizes 64/128 and 2 layers.
   - Transformer: try `d_model` 64/128, 2-4 encoder layers, dropout 0.1-0.3.
   - Conv-Transformer: add multi-scale Conv1D kernels such as 3, 7, 15 and
     concatenate them before the Transformer.

4. Better loss and horizon handling:
   - try Huber loss during training while still reporting MSE/MAE;
   - add horizon-weighted loss so early and late forecast days are both learned;
   - consider direct multi-output prediction first, then compare an
     encoder-decoder variant if time allows.

5. Feature engineering:
   - add calendar features: month, day-of-week, weekend flag, day-of-year
     sine/cosine;
   - add rolling means and rolling std over 7, 30, and 90 days;
   - include real weather features when available.

Do not inflate results or cherry-pick only successful seeds. If the improved
model is novel but not best-performing, explain why in the discussion, as the
PDF allows novelty and analysis to matter.

## Completion Checklist

- `python -m src.data` works from a clean checkout.
- GPU verification shows `torch.cuda.is_available() == True`.
- Full experiments complete:
  - 3 models x 2 horizons x 5 seeds.
- Final metrics saved:
  - `outputs/metrics/full_metrics.csv`
  - `outputs/metrics/full_summary.csv`
  - `outputs/metrics/full_summary.json`
- Figures saved:
  - one prediction-vs-ground-truth plot per model and horizon;
  - optional combined comparison plots.
- Report complete:
  - `report/report.md`
  - `report/report.pdf` if export tooling is available.
- README updated with final GPU commands.
- GitHub link inserted into the report if the user provides or creates a repo.
- References included for:
  - UCI dataset;
  - Meteo-France/data.gouv.fr climate data;
  - LSTM;
  - Transformer;
  - any improved-model inspiration.
