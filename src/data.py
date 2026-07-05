"""Download, aggregate, and window the household power data.

This module intentionally avoids pandas/sklearn so it can run in the existing
``yolov8`` conda environment used for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

from .constants import (
    CALENDAR_COLUMNS,
    DAILY_POWER_CSV_PATH,
    FEATURE_COLUMNS,
    INPUT_DAYS,
    LOCAL_TEST_CANDIDATES,
    LOCAL_TRAIN_CANDIDATES,
    METEO_BASE_URL,
    METEO_DEPT_STATIONS,
    METEO_FALLBACK_STATION,
    METEO_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    UCI_FALLBACK_URL,
    UCI_POWER_URL,
    UCI_RAW_TXT_PATH,
    UCI_ZIP_PATH,
    WEATHER_COLUMNS,
    WEATHER_RAW_CSV_PATH,
    ensure_project_dirs,
)

RAW_NUMERIC_COLUMNS = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)

DAILY_COLUMNS = (
    "date",
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "sub_metering_remainder",
    *WEATHER_COLUMNS,
    *CALENDAR_COLUMNS,
)


def _download_file(urls: Iterable[str], destination: Path, timeout: int = 120) -> Path:
    ensure_project_dirs()
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    last_error: Exception | None = None
    for url in urls:
        try:
            tmp_path = destination.with_suffix(destination.suffix + ".part")
            existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
            mode = "ab" if existing_size else "wb"

            print(f"Downloading {url}")
            if existing_size:
                print(f"Resuming from {existing_size} bytes")

            with requests.get(url, stream=True, timeout=(10, timeout), headers=headers) as response:
                response.raise_for_status()
                if response.status_code == 200 and existing_size:
                    mode = "wb"
                with tmp_path.open(mode) as out_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out_file.write(chunk)
            if zipfile.is_zipfile(tmp_path):
                tmp_path.replace(destination)
                return destination
            last_error = RuntimeError(f"Downloaded file is not a valid zip: {tmp_path}")
        except Exception as exc:  # pragma: no cover - exercised by network state
            last_error = exc
    raise RuntimeError(f"Could not download {destination.name}: {last_error}")


def download_uci_power() -> Path:
    """Download the UCI household power zip if needed."""
    return _download_file((UCI_FALLBACK_URL, UCI_POWER_URL), UCI_ZIP_PATH)


def extract_uci_power() -> Path:
    """Extract ``household_power_consumption.txt`` if needed."""
    ensure_project_dirs()
    if UCI_RAW_TXT_PATH.exists() and UCI_RAW_TXT_PATH.stat().st_size > 0:
        return UCI_RAW_TXT_PATH

    download_uci_power()
    with zipfile.ZipFile(UCI_ZIP_PATH) as archive:
        txt_members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not txt_members:
            raise RuntimeError(f"No text file found in {UCI_ZIP_PATH}")
        member = txt_members[0]
        with archive.open(member) as source, UCI_RAW_TXT_PATH.open("wb") as target:
            target.write(source.read())
    return UCI_RAW_TXT_PATH


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if value == "" or value == "?":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_weather_by_month(path: Path = WEATHER_RAW_CSV_PATH) -> dict[str, dict[str, float]]:
    """Read optional monthly weather CSV keyed by YYYY-MM."""
    if not path.exists():
        return {}

    month_weather: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        sample = file_obj.read(4096)
        file_obj.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(file_obj, delimiter=delimiter)
        for row in reader:
            month_key = None
            for key in ("month", "date", "DATE", "AAAAMM", "YYYYMM"):
                value = (row.get(key) or "").strip()
                if not value:
                    continue
                digits = "".join(ch for ch in value if ch.isdigit())
                if len(digits) >= 6:
                    month_key = f"{digits[:4]}-{digits[4:6]}"
                    break
            if month_key is None:
                year = (row.get("year") or row.get("YEAR") or row.get("ANNEE") or "").strip()
                month = (row.get("month") or row.get("MONTH") or row.get("MOIS") or "").strip()
                if year and month:
                    month_key = f"{int(float(year)):04d}-{int(float(month)):02d}"
            if month_key is None:
                continue

            values = {}
            for column in WEATHER_COLUMNS:
                parsed = _parse_float(row.get(column, ""))
                values[column] = 0.0 if parsed is None else parsed
            month_weather[month_key] = values
    return month_weather


def _download_meteo_resource(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)
    tmp_path.replace(destination)
    return destination


def download_meteo_weather(
    station_id: str = METEO_FALLBACK_STATION,
    destination: Path = WEATHER_RAW_CSV_PATH,
) -> Path:
    """Download monthly climate data from Meteo-France for a specific station.

    Data is fetched from data.gouv.fr / Meteo-France's S3 bucket, filtered to a
    single station, and cached locally as weather_monthly.csv.
    """
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    dept = station_id[:2]
    periods = [
        f"MENSQ_{dept}_previous-1950-2024.csv.gz",
        f"MENSQ_{dept}_latest-2025-2026.csv.gz",
    ]
    raw_dir = RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    station_rows: list[dict[str, str]] = []
    for period_file in periods:
        url = f"{METEO_BASE_URL}{METEO_PATH}{period_file}"
        gz_path = raw_dir / period_file
        if not gz_path.exists():
            print(f"Downloading weather data: {url}")
            _download_meteo_resource(url, gz_path)
        else:
            print(f"Using cached weather data: {gz_path}")

        with gzip.open(gz_path, "rt", encoding="utf-8-sig", newline="") as file_obj:
            reader = csv.DictReader(file_obj, delimiter=";")
            for row in reader:
                if row.get("NUM_POSTE", "").strip() == station_id:
                    month = (row.get("AAAAMM") or "").strip()
                    if not month:
                        continue
                    station_rows.append(row)

    if not station_rows:
        print(f"Station {station_id} not found in Meteo-France data; weather will be zero-filled.")
        return destination

    station_rows.sort(key=lambda r: r.get("AAAAMM", ""))
    weather_data: dict[str, dict[str, float]] = {}
    for row in station_rows:
        month_val = row["AAAAMM"]
        if len(month_val) >= 6:
            month_key = f"{month_val[:4]}-{month_val[4:6]}"
        else:
            continue
        values = {}
        for column in WEATHER_COLUMNS:
            parsed = _parse_float(row.get(column, ""))
            values[column] = 0.0 if parsed is None else float(parsed)
        weather_data[month_key] = values

    with destination.open("w", encoding="utf-8", newline="") as file_obj:
        fieldnames = ["month"] + list(WEATHER_COLUMNS)
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for month_key in sorted(weather_data):
            row_out = {"month": month_key}
            row_out.update(weather_data[month_key])
            writer.writerow(row_out)

    print(f"Saved {len(weather_data)} month records for station {station_id} to {destination}")
    return destination


def aggregate_raw_to_daily(raw_path: Path = UCI_RAW_TXT_PATH) -> list[dict[str, float | str]]:
    """Aggregate minute-level UCI records to daily rows."""
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "global_active_power": 0.0,
            "global_reactive_power": 0.0,
            "voltage": 0.0,
            "global_intensity": 0.0,
            "sub_metering_1": 0.0,
            "sub_metering_2": 0.0,
            "sub_metering_3": 0.0,
            "sub_metering_remainder": 0.0,
            "voltage_count": 0.0,
            "global_intensity_count": 0.0,
            "row_count": 0.0,
        }
    )

    with raw_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter=";")
        for row in reader:
            date_text = row.get("Date", "")
            try:
                day = datetime.strptime(date_text, "%d/%m/%Y").date().isoformat()
            except ValueError:
                continue

            values = {column: _parse_float(row.get(column, "")) for column in RAW_NUMERIC_COLUMNS}
            if values["Global_active_power"] is None:
                continue

            day_sums = sums[day]
            gap = float(values["Global_active_power"])
            grp = float(values["Global_reactive_power"] or 0.0)
            voltage = values["Voltage"]
            intensity = values["Global_intensity"]
            sm1 = float(values["Sub_metering_1"] or 0.0)
            sm2 = float(values["Sub_metering_2"] or 0.0)
            sm3 = float(values["Sub_metering_3"] or 0.0)

            day_sums["global_active_power"] += gap
            day_sums["global_reactive_power"] += grp
            day_sums["sub_metering_1"] += sm1
            day_sums["sub_metering_2"] += sm2
            day_sums["sub_metering_3"] += sm3
            day_sums["sub_metering_remainder"] += (gap * 1000.0 / 60.0) - (sm1 + sm2 + sm3)
            day_sums["row_count"] += 1.0

            if voltage is not None:
                day_sums["voltage"] += float(voltage)
                day_sums["voltage_count"] += 1.0
            if intensity is not None:
                day_sums["global_intensity"] += float(intensity)
                day_sums["global_intensity_count"] += 1.0

    month_weather = _read_weather_by_month()
    daily_rows: list[dict[str, float | str]] = []
    for day in sorted(sums):
        item = sums[day]
        voltage_count = max(item["voltage_count"], 1.0)
        intensity_count = max(item["global_intensity_count"], 1.0)
        row: dict[str, float | str] = {
            "date": day,
            "global_active_power": item["global_active_power"],
            "global_reactive_power": item["global_reactive_power"],
            "voltage": item["voltage"] / voltage_count,
            "global_intensity": item["global_intensity"] / intensity_count,
            "sub_metering_1": item["sub_metering_1"],
            "sub_metering_2": item["sub_metering_2"],
            "sub_metering_3": item["sub_metering_3"],
            "sub_metering_remainder": item["sub_metering_remainder"],
        }
        weather = month_weather.get(day[:7], {})
        for column in WEATHER_COLUMNS:
            row[column] = float(weather.get(column, 0.0))
        dt = datetime.strptime(day, "%Y-%m-%d")
        month_val = float(dt.month)
        dow_val = float(dt.weekday())
        row["month"] = month_val
        row["day_of_week"] = dow_val
        row["is_weekend"] = 1.0 if dow_val >= 5 else 0.0
        row["day_sin"] = math.sin(2.0 * math.pi * float(dt.timetuple().tm_yday) / 366.0)
        row["day_cos"] = math.cos(2.0 * math.pi * float(dt.timetuple().tm_yday) / 366.0)
        daily_rows.append(row)
    return daily_rows


def write_daily_csv(rows: list[dict[str, float | str]], path: Path = DAILY_POWER_CSV_PATH) -> Path:
    ensure_project_dirs()
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=DAILY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def prepare_daily_data(force: bool = False) -> Path:
    """Ensure the processed daily CSV exists and return its path."""
    ensure_project_dirs()
    try:
        download_meteo_weather()
    except Exception as exc:
        print(f"Weather download failed, using zero-fill fallback: {exc}")
    if DAILY_POWER_CSV_PATH.exists() and not force:
        return DAILY_POWER_CSV_PATH
    raw_path = extract_uci_power()
    rows = aggregate_raw_to_daily(raw_path)
    if len(rows) < INPUT_DAYS * 2:
        raise RuntimeError(f"Too few daily rows after aggregation: {len(rows)}")
    return write_daily_csv(rows)


def _read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        sample = file_obj.read(4096)
        file_obj.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(file_obj, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        numeric_columns = [name for name in fieldnames if name != "date"]
        rows = []
        for row in reader:
            values = []
            for column in numeric_columns:
                parsed = _parse_float(row.get(column, ""))
                values.append(float("nan") if parsed is None else parsed)
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    array = np.asarray(rows, dtype=np.float32)
    return numeric_columns, fill_missing_array(array)


def load_daily_matrix(path: Path = DAILY_POWER_CSV_PATH) -> tuple[list[str], np.ndarray]:
    """Load processed daily data and return selected feature columns."""
    if not path.exists():
        prepare_daily_data()
    columns, array = _read_numeric_csv(path)
    index = {name: i for i, name in enumerate(columns)}
    missing = [column for column in FEATURE_COLUMNS if column not in index]
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    selected = array[:, [index[column] for column in FEATURE_COLUMNS]]
    return list(FEATURE_COLUMNS), selected.astype(np.float32)


def fill_missing_array(array: np.ndarray) -> np.ndarray:
    """Fill NaNs with column means, using zeros for all-NaN columns."""
    result = array.astype(np.float32, copy=True)
    for column_idx in range(result.shape[1]):
        column = result[:, column_idx]
        mask = np.isfinite(column)
        fill_value = float(column[mask].mean()) if mask.any() else 0.0
        column[~mask] = fill_value
    return result


def build_windows(
    matrix: np.ndarray,
    input_days: int = INPUT_DAYS,
    output_days: int = 90,
    target_column: str = TARGET_COLUMN,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows from daily feature matrix."""
    feature_columns = list(feature_columns)
    target_idx = feature_columns.index(target_column)
    total_window = input_days + output_days
    count = matrix.shape[0] - total_window + 1
    if count <= 0:
        raise RuntimeError(
            f"Not enough rows ({matrix.shape[0]}) for input={input_days}, output={output_days}"
        )

    x = np.empty((count, input_days, matrix.shape[1]), dtype=np.float32)
    y = np.empty((count, output_days), dtype=np.float32)
    for idx in range(count):
        x[idx] = matrix[idx : idx + input_days]
        y[idx] = matrix[idx + input_days : idx + total_window, target_idx]
    return x, y


def split_and_scale_windows(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    max_train_windows: int | None = None,
    target_feature_index: int = 0,
) -> dict[str, np.ndarray]:
    """Sequential train/test split with train-only standardization."""
    split_idx = max(1, int(math.floor(len(x) * train_ratio)))
    split_idx = min(split_idx, len(x) - 1)
    train_x = x[:split_idx]
    train_y = y[:split_idx]
    test_x = x[split_idx:]
    test_y = y[split_idx:]

    if max_train_windows is not None:
        train_x = train_x[:max_train_windows]
        train_y = train_y[:max_train_windows]
        test_x = test_x[: max(1, min(max_train_windows // 4, len(test_x)))]
        test_y = test_y[: max(1, min(max_train_windows // 4, len(test_y)))]

    feature_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True)
    feature_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0

    target_mean = feature_mean.reshape(-1)[target_feature_index].astype(np.float32)
    target_std = feature_std.reshape(-1)[target_feature_index].astype(np.float32)
    if target_std < 1e-6:
        target_std = np.float32(1.0)

    return {
        "train_x": ((train_x - feature_mean) / feature_std).astype(np.float32),
        "train_y": ((train_y - target_mean) / target_std).astype(np.float32),
        "test_x": ((test_x - feature_mean) / feature_std).astype(np.float32),
        "test_y": ((test_y - target_mean) / target_std).astype(np.float32),
        "test_y_raw": test_y.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "target_mean": np.asarray(target_mean, dtype=np.float32),
        "target_std": np.asarray(target_std, dtype=np.float32),
    }


def find_local_train_test() -> tuple[Path | None, Path | None]:
    train = next((path for path in LOCAL_TRAIN_CANDIDATES if path.exists()), None)
    test = next((path for path in LOCAL_TEST_CANDIDATES if path.exists()), None)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare daily household power data.")
    parser.add_argument("--force", action="store_true", help="Regenerate processed data.")
    args = parser.parse_args()

    train_path, test_path = find_local_train_test()
    if train_path and test_path:
        print(f"Found local train/test files: {train_path} | {test_path}")
    processed = prepare_daily_data(force=args.force)
    columns, matrix = load_daily_matrix(processed)
    print(f"Processed daily CSV: {processed}")
    print(f"Rows: {matrix.shape[0]}, feature columns: {len(columns)}")


if __name__ == "__main__":
    main()
