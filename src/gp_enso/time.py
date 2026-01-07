from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_REFERENCE_TIME = pd.Timestamp("1870-08-01")

def dates_to_year_index(
    dates: pd.datetimeIndex | pd.Series,
    reference_time: pd.Timestamp = DEFAULT_REFERENCE_TIME,
) -> np.ndarray:
    """
    Convert datetimes to a numeric time index in years since reference_time
    Notebook: dates_to_idx()
    """
    dt = pd.to_datetime(dates)
    t = (dt - reference_time) / pd.Timedelta(365, "D")
    return np.asarray(t, dtype=float)