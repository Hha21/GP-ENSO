from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from bokeh.plotting import figure, show
from bokeh.models import Span

from gp_enso.io import prepare_enso_dataframe
from gp_enso.explore import plot_autocorrelation, plot_periodogram_years
from gp_enso.gp_model import build_quasiperiodic_gp_model, fit_map
from gp_enso.forecast import make_monthly_date_grid, predict_gp, draw_paths
from gp_enso.plot import plot_gp_forecast


def plot_timeseries_bokeh(df):
    p = figure(
        x_axis_type="datetime",
        title="NINO 3.4 Index over time (smoothed)",
        width=800,
        height=450,
    )
    p.yaxis.axis_label = "NINA3.4"
    p.xaxis.axis_label = "Date"

    zeroline = Span(location=0, dimension="width", line_color="red", line_dash="dashed", line_width=2)
    p.add_layout(zeroline)

    p.line(df.index, df["NINA34_smoothed"], line_width=2, line_color="black", alpha=0.6)
    show(p)


def main():

    prepared = prepare_enso_dataframe(REPO_ROOT / "data" / "nino34.long.anom.csv")
    df = prepared.df

    t = (df["t"].values[:, None])
    y = df["y_n"].values

    model, gp = build_quasiperiodic_gp_model(t, y)
    mp = fit_map(model)

    dates = make_monthly_date_grid(start="1870-01-01", end="2040-08-01", freq="MS") # Month Start
    pred = predict_gp(model, gp, mp, dates)

    # rescale back to original units
    mu_sc = pred.mu * prepared.value_std + prepared.first_value
    cov_sc = pred.cov * (prepared.value_std ** 2)

    samples = draw_paths(mu_sc, cov_sc, draws=5, seed=1)

    p = plot_gp_forecast(
        dates=pred.dates,
        mu=mu_sc,
        cov=cov_sc,
        samples=samples,
        df_obs=df,
        obs_col=prepared.smooth_col,  # or prepared.value_col
        split_date="2025-08-01",
        title="ENSO GP forecast",
    )
    show(p)


if __name__ == "__main__":
    main()