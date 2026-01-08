from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import Span

def plot_gp_forecast(
    dates: pd.DatetimeIndex,
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    samples: np.ndarray | None,
    df_obs: pd.DataFrame,
    obs_col: str,
    split_date: str = "2025-08-01",
    title: str = "GP forecast"
):
    p = figure(x_axis_type="datetime", width=900, height=360, title=title)
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = obs_col

    # plot mean and 2σ region of total prediction
    # scale mean and var
    mu = np.asarray(mu).reshape(-1)
    cov = np.asarray(cov)
    sd = np.sqrt(np.diag(cov))

    upper = mu + 2 * sd
    lower = mu - 2 * sd

    band_x = np.append(dates, dates[::-1])
    band_y = np.append(lower, upper[::-1])

    p.line(dates, mu, line_width=2, line_color="firebrick", legend_label="Total fit")
    p.patch(band_x, band_y, color="firebrick", alpha=0.6, line_color="white", legend_label="±2σ")

    if samples is not None:
        for i in range(samples.shape[0]):
            p.line(dates, samples[i, :], alpha=0.25, line_width=1)

    p.scatter(
        df_obs.index,
        df_obs[obs_col],
        marker="circle",
        line_color="black",
        alpha=0.15,
        size=4,
        legend_label="Observed",
    )

    predline = Span(location=pd.to_datetime(split_date), dimension="height", line_dash="dashed", line_width=2)
    p.add_layout(predline)
    p.legend.location = "bottom_right"
    return p

def plot_timeseries(
    x,
    y,
    *,
    title: str,
    y_label: str,
    width: int = 800,
    height: int = 450,
    zero_line: bool = False,
):
    p = figure(x_axis_type="datetime", title=title, width=width, height=height)
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = y_label
    if zero_line:
        p.add_layout(Span(location=0, dimension="width", line_dash="dashed", line_width=2))
    p.line(x, y, line_width=2, alpha=0.6)
    return p