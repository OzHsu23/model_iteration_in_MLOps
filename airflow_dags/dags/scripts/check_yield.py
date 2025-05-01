import os
import json
import warnings
import pandas as pd
from schemas import FullConfig, MonitorYieldConfig  # 保留兩個 class 引用

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


def check_yield(log_path: str, config_path: str, flag_path: str) -> None:
    # Load full nested config
    with open(config_path, "r") as f:
        raw_config = json.load(f)

    full_config = FullConfig(**raw_config)

    # Override from Airflow params for absolute paths
    monitor_cfg: MonitorYieldConfig = full_config.monitor.copy(update={
        "log_path": log_path,
        "flag_path": flag_path
    })

    if not os.path.exists(monitor_cfg.log_path):
        raise FileNotFoundError(f"Inference log not found at {monitor_cfg.log_path}")

    df = pd.read_csv(monitor_cfg.log_path)
    if df.empty:
        raise ValueError("Inference log is empty.")

    # Take recent N samples
    recent_df = df.tail(monitor_cfg.recent_window)
    if recent_df.empty or recent_df["pred"].isnull().all():
        warnings.warn("Not enough recent valid samples to compute yield. Assuming yield = 0.")
        current_yield = 0.0
    else:
        current_yield = compute_yield(recent_df)

    # If no threshold is provided, compute it from historical data
    if monitor_cfg.threshold is None:
        historical_yield = compute_yield(df)
        threshold = max(historical_yield - monitor_cfg.delta, 0.6)
        print(f"[INFO] No threshold provided. Using historical yield: {historical_yield:.3f} - delta: {monitor_cfg.delta:.3f} = {threshold:.3f}")   
    else:
        threshold = monitor_cfg.threshold
        print(f"[INFO] Current yield = {current_yield:.3f}, Threshold = {threshold:.3f}")

    # Trigger retrain if current yield is below threshold
    if current_yield < threshold:
        os.makedirs(os.path.dirname(monitor_cfg.flag_path), exist_ok=True)
        with open(monitor_cfg.flag_path, "w") as f:
            f.write("trigger retrain")
        print(f"[INFO] Triggering retrain: Current yield {current_yield:.3f} < Threshold {threshold:.3f}")
    else:
        if os.path.exists(monitor_cfg.flag_path):
            os.remove(monitor_cfg.flag_path)
