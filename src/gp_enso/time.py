from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_REFERENCE_TIME = pd.Timestamp("1870-08-01")

def dates_to_year_index(
    dates: pd.DatetimeIndex | pd.Series,
    reference_time: pd.Timestamp = DEFAULT_REFERENCE_TIME,
) -> np.ndarray:
    """
    Convert datetimes to a numeric time index in years since reference_time
    Notebook: dates_to_idx()
    """
    dt = pd.to_datetime(dates)
    t = (dt - reference_time) / pd.Timedelta(365, "D")
    return np.asarray(t, dtype=float)

def dates_to_index(
    dates: pd.DatetimeIndex | pd.Series,
    reference_time: str | pd.Timestamp,
    unit: str = "D",
    scale: float = 365.0,
) -> np.ndarray:
    """
    Convert datetimes to numeric index.

    t = (dates - reference_time) / pd.Timedelta(scale, unit)

    Examples:
      - years since 1870-08-01: unit="D", scale=365
      - 0.01-day units since 2024-01-01: unit="D", scale=0.01
    """
    dt = pd.to_datetime(dates)
    ref = pd.to_datetime(reference_time)
    t = (dt - ref) / pd.Timedelta(scale, unit)
    return np.asarray(t, dtype=float)