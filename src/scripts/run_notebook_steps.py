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

    prepared = prepare_enso_dataframe(REPO_ROOT / "data" / "nino34.long.anom.csv", rolling_window=3)
    df = prepared.df

    # Notebook plot block
    # plot_timeseries_bokeh(df)

    # Notebook ACF block (uses raw NINO34 in notebook)
    dominant_period = plot_periodogram_years(df["NINA34"].to_numpy())
    print(f"Dominant period: {dominant_period:.2f} years")

    plot_autocorrelation(df["NINA34"].to_numpy(), lags=100)


if __name__ == "__main__":
    main()