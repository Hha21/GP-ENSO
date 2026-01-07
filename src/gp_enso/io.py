from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .time import dates_to_year_index

@dataclass(frozen=True)
class ENSOPrepared:
    """
    Prepared dataset:
        - datetime index
        - smoothed target column
        - numeric time index 't' in yrs since ref
        - normalised series 'y_n'
        - normalisation parameters
    """
    df: pd.DataFrame
    value_col: str
    smooth_col: str
    first_value: float
    value_std: float

def load_nino34_csv(
    csv_path: str | Path,
) -> pd.DataFrame:
    """
    Load NOAA CSV.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df.columns = ["Date", "NINA34"]

    # if date_col not in df.columns:
    #     raise ValueError(f"Expected a '{date_col}' column in {csv_path}, got columns: {list(df.columns)}")
    
    # candidate_cols = [c for c in df.columns if c.upper() in {"NINA34"}]
    # if not candidate_cols:
    #     raise ValueError(f"Expected a NINO34/NINA34 column in {csv_path}, got columns: {list(df.columns)}")

    # value_col = candidate_cols[0]

    return df

def clean_and_index(
    df: pd.DataFrame,
    missing_sentinel: float = -9999.0,
) -> pd.DataFrame:
    """
    Replace sentinel with NaN, drop missing, set datetime index from 'Date'.
    """
    out = df.copy()
    out["NINA34"] = out["NINA34"].replace(missing_sentinel, np.nan)
    out = out.dropna(subset=["NINA34"])
    out.index = pd.to_datetime(out["Date"])
    return out

def add_rolling_smooth(
    df: pd.DataFrame,
    window: int = 3,
    center: bool = True,
    smooth_col: str = "NINA34_smoothed"
) -> pd.DataFrame:
    """
    Add rolling mean smooth, drop NaNs introduced by smoothing.
    """
    out = df.copy()
    out[smooth_col] = out["NINA34"].rolling(window=window, center=center).mean()
    out = out.dropna(subset=[smooth_col])
    return out

def normalise_series(
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Normalisation, returns tuple of:
        - y[0]
        - y_std
        - (y - y[0]) / y_std
    """
    y = np.asarray(y, dtype=float)
    first_value = float(y[0])
    y_std = float(np.std(y))
    if y_std == 0:
        raise ValueError("Standard deviation is zero; cannot normalise.")
    y_n = (y - first_value) / y_std
    return y_n, first_value, y_std

def prepare_enso_dataframe(
    csv_path: str | Path,
    *,
    rolling_window: int = 3,
    reference_time: pd.Timestamp | None = None,
) -> ENSOPrepared:
    """
    Function to reproduce notebook state up to t and y_n creation
    """
    df0 = load_nino34_csv(csv_path)
    df1 = clean_and_index(df0)
    df2 = add_rolling_smooth(df1, window=rolling_window, smooth_col="NINA34_smoothed")

    t = dates_to_year_index(df2.index, reference_time=reference_time) if reference_time is not None else dates_to_year_index(df2.index)
    y = df2["NINA34_smoothed"].to_numpy()
    y_n, first_value, y_std = normalise_series(y)

    df3 = df2.assign(t=t, y_n=y_n)

    return ENSOPrepared(
        df = df3,
        value_col="NINA34",
        smooth_col="NINA34_smoothed",
        first_value=first_value,
        value_std=y_std,
    )
