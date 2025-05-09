# dags/scripts/check_yield.py

import os
import json
import warnings
import pandas as pd
from schemas import FullConfig, MonitorYieldConfig

def compute_accuracy(df: pd.DataFrame) -> float:
    """Compute prediction accuracy: pred == label"""
    df = df.dropna(subset=["pred", "label"])
    if df.empty:
        return 0.0
    return (df["pred"] == df["label"]).mean()

def compute_yield(df: pd.DataFrame, ok_label: str = "OK") -> float:
    """Compute yield = proportion of samples predicted as OK"""
    df = df.dropna(subset=["pred"])
    if df.empty:
        return 0.0
    return (df["pred"] == ok_label).mean()

def check_yield(config: FullConfig) -> None:
    """
    Check current yield based on the log path defined in the config.
    """
    monitor_cfg: MonitorYieldConfig = config.monitor
    log_path = monitor_cfg.log_path

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Inference log not found at {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError("Inference log is empty.")

    recent_df = df.tail(monitor_cfg.recent_window)
    if recent_df.empty or recent_df["pred"].isnull().all():
        warnings.warn("Not enough recent valid samples to compute yield. Assuming yield = 0.")
        current_yield = 0.0
    else:
        current_yield = compute_yield(recent_df)

    if monitor_cfg.yield_threshold is None:
        historical_yield = compute_yield(df)
        threshold = max(historical_yield - monitor_cfg.yield_drop_tolerance, 0.6)
        print(f"[INFO] No threshold provided. Using historical yield: {historical_yield:.3f} - drop_tol: {monitor_cfg.yield_drop_tolerance:.3f} = {threshold:.3f}")
    else:
        threshold = monitor_cfg.yield_threshold
        print(f"[INFO] Current yield = {current_yield:.3f}, Threshold = {threshold:.3f}")

    if current_yield < threshold:
        os.makedirs(os.path.dirname(monitor_cfg.flag_path), exist_ok=True)
        with open(monitor_cfg.flag_path, "w") as f:
            f.write("trigger retrain")
        print(f"[INFO] Triggering retrain: Current yield {current_yield:.3f} < Threshold {threshold:.3f}")
    else:
        if os.path.exists(monitor_cfg.flag_path):
            os.remove(monitor_cfg.flag_path)
            print(f"[INFO] Yield is sufficient: {current_yield:.3f} >= {threshold:.3f}. Flag cleared.")
