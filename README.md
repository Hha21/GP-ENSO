# GP-ENSO

El Niño-Southern Oscillation

## Project structure

```bash
├── data/
|   ├── nino12.long.anom.csv
|   └── nino34.long.anom.csv
├── src/
|   ├── gp_enso/
|   |   ├── __init__.py
|   |   ├── io.py       # loading + cleaning + smoothing + normalisation
|   |   ├── time.py     # time index helpers (date -> numeric time)
|   |   └── explore.py  # exploratory diagnostics (ACF, FFT/periodogram)
|   └── scripts/
|       └── run_notebook_steps.py # reproduce notebook
├── environment.yml # conda environment specification
├── GPENSO.ipynb # original notebook (reference)
└── README.md
```

## Environment setup (conda)

We use conda for reproducibility.

### Create the environment
From the repo root:

```bash
conda env create -f environment.yml
conda activate gp_enso
```

### macOS / Linux

From the repo root:
```
conda activate gp_enso
export PYTHONPATH="$(pwd)/src"
python src/scripts/run_notebook_steps.py
```

### Windows PowerShell
From the repo root:
```bash
conda activate gp_enso
$env:PYTHONPATH = "$(Get-Location)\src"
python src\scripts\run_notebook_steps.py
```
