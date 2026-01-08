from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import pymc as pm

from .time import dates_to_year_index

@dataclass(frozen=True)
class GPPrediction:
    dates: pd.DatetimeIndex
    tnew: np.ndarray            # (M,1)
    mu: np.ndarray              # (M,)
    cov: np.ndarray             # (M,M)

def make_monthly_date_grid(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str = "MS",
) -> pd.DatetimeIndex:
    return pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq=freq)

def predict_gp(
    model: pm.Model,
    gp: pm.gp.Marginal,
    mp: dict,
    dates: pd.DatetimeIndex,
) -> GPPrediction:

    tnew = dates_to_year_index(dates)[:, None]
    print("Sampling GP predictions ...")

    with model:
        mu, cov = gp.predict(tnew, point=mp)
    
    mu = np.asarray(mu, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)

    return GPPrediction(dates=dates, tnew=tnew, mu=mu, cov=cov)

def draw_paths(
    mu: np.ndarray,
    cov: np.ndarray,
    draws: int = 5,
    seed: int = -1,
) -> np.ndarray:
    """
    Draw sample functions from N(mu, cov).
    Returns shape (draws, M).
    """
    rv = pm.MvNormal.dist(mu=mu, cov=cov)
    return pm.draw(rv, draws=draws, random_seed=seed)