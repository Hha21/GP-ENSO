from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.fft import fft, fftfreq

def plot_autocorrelation(
    y: np.ndarray,
    lags: int = 100,
    title: str = "Autocorrelation Function",
):
    y = np.asarray(y, dtype=float)
    plt.figure(figsize=(12, 5))
    plot_acf(y, lags=lags)
    plt.xlabel("Lag (months)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_periodogram_years(
    y: np.ndarray,
    sample_spacing_years: float = 1.0 / 12.0,
    xlim_years: float = 10.0,
):
    y = np.asarray(y, dtype=float)
    n = len(y)

    fft_vals = fft(y)
    fft_power = np.abs(fft_vals) ** 2

    freqs = fftfreq(n, d=sample_spacing_years)
    mask = freqs > 0

    freqs = freqs[mask]
    fft_power = fft_power[mask]

    periods = 1.0 / freqs
    plt.figure(figsize=(10, 5))
    plt.plot(periods, fft_power)
    plt.xlabel("Period (years)")
    plt.ylabel("Power")
    plt.xlim(0, xlim_years)
    plt.title("Periodogram")
    plt.tight_layout()
    plt.show()

    peak_idx = int(np.argmax(fft_power))
    return float(periods[peak_idx])