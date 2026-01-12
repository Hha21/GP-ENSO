from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import requests

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

def prepare_sunspots(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["Sunspot_Number"])
    out.index = pd.to_datetime(out["Date"])
    return out

def load_sunspots(
    url: str = "https://www.sidc.be/SILSO/INFO/snmtotcsv.php",
) -> pd.DataFrame:
    """
    Load yearly mean total sunspot number (SILSO)
    """
    df = pd.read_csv(
        url,
        sep = ";",
        header = None,
        names = ["Year", "Month", "Date", "Sunspot_Number", "StdDev", "Observations", "Definitive"],
    )

    df["Date"] = pd.to_datetime(df["Year"].astype(int).astype(str), format="%Y")
    return df

def download_noaa_tides(
    station_id: str,
    start_date: str,
    end_date: str,
    product: str = "water_level",
) -> pd.DataFrame:
    """
    Download tide/current data from NOAA Tides & Currents API.
    Dates are YYYYMMDD strings.
    Returns a DataFrame with columns ['time', 'water_level'] (floats) and datetime 'time'.
    """
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "station": station_id,
        "product": product,
        "begin_date": start_date,
        "end_date": end_date,
        "datum": "MLLW",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "research",
    }

    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if "data" not in payload:
        raise RuntimeError(f"NOAA API returned no 'data'. Payload keys: {list(payload.keys())}. Error: {payload.get('error')}")

    df = pd.DataFrame(payload["data"])
    # NOAA uses 't' and 'v' strings
    df["time"] = pd.to_datetime(df["t"])
    df["water_level"] = pd.to_numeric(df["v"], errors="coerce")
    df = df[["time", "water_level"]].dropna()
    df = df.sort_values("time").reset_index(drop=True)
    return df

def subsample(df: pd.DataFrame, step: int = 10) -> pd.DataFrame:
    """
    Take every `step`th row (like df.iloc[::10]).
    """
    return df.iloc[::step].copy()

def normalise_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, float, float]:
    """
    Add y_n column from `col` using notebook scheme.
    Returns (df_with_y_n, first_value, y_std)
    """
    y = df[col].to_numpy(dtype=float)
    first = float(y[0])
    std = float(np.std(y))
    if std == 0:
        raise ValueError("Std is zero; cannot normalise.")
    y_n = (y - first) / std
    out = df.copy()
    out["y_n"] = y_n
    return out, first, std