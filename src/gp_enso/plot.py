from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf

from . import config

def plot_multi_input_gp(
    *,
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    y_test: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    split_t: float | None = None,
    title: str = "Multi-Input GP: Predictions on Test Set",
    ylabel: str = "Ni\u00f1o 3.4 Anomaly (\u00b0C)",
):  

    t_train = np.asarray(t_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    t_test = np.asarray(t_test, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    mean_pred = np.asarray(mean_pred, dtype=float).reshape(-1)
    std_pred = np.asarray(std_pred, dtype=float).reshape(-1)

    if split_t is None and len(t_train) > 0:
        split_t = float(t_train[-1])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t_train, y_train, "k-", alpha=0.7, linewidth=0.8, label="Training Data")
    ax.plot(t_test, y_test, "gray", alpha=0.7, linewidth=1.5, label="Actual (Test)", linestyle="--")
    ax.plot(t_test, mean_pred, "r-", linewidth=2, label="GP Mean Prediction")

    ax.fill_between(
        t_test,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        alpha=0.3,
        color="red",
        label="95% CI",
    )

    if split_t is not None:
        ax.axvline(x=split_t, color="blue", linestyle=":", linewidth=2, label="Train/Test Split")

    ax.set_xlabel("Time (years since start)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    
    save_dir = config.PLOT_DIR / "GP-Prediction.png"
    plt.savefig(save_dir)

