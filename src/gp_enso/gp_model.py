from __future__ import annotations

import numpy as np
import pymc as pm

def build_quasiperiodic_gp_model(
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[pm.Model, pm.gp.Marginal]:
    """
    GP Model:
    - Quasi-periodic kernel: ExpQuad * Periodic
    - Noise kernel: Matern32
    - Gaussian likelihood with sigma parameter
    - Uses pm.gp.Marginal(marginal_likelihood)

    Inputs:
      t: shape (N,) or (N,1) numeric time index (years)
      y: shape (N,) normalised series (e.g. y_n)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    with pm.Model() as model:

        # Quasi-Periodic
        n_per1 = pm.HalfCauchy("n_per1", beta=1.5, initval=0.5)
        l_per1 = pm.Gamma("l_per1", alpha=1, beta=2)
        period_1 = pm.Gamma("period_1", alpha=8, beta=2)
        l_exp = pm.Gamma("l_exp", alpha=2, beta=1)

        cov_per1 = (
            n_per1**2
            * pm.gp.cov.ExpQuad(1, l_exp)
            * pm.gp.cov.Periodic(1, period_1, l_per1)
        )

        # Noise Model
        sigma = pm.HalfNormal("sigma", sigma=0.25, initval=0.1)
        n_noise = pm.HalfNormal("n_noise", sigma=0.5, initval=0.05)
        l_noise = pm.Gamma("l_noise", alpha=2, beta=1)

        cov_noise = (
            n_noise**2
            * pm.gp.cov.Matern32(1, l_noise)
        )

        cov_total = cov_per1 + cov_noise

        gp = pm.gp.Marginal(cov_func=cov_total)

        # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
        gp.marginal_likelihood("y", X=t, y=y, sigma=sigma)

    return model, gp

def fit_map(model: pm.Model) -> dict:
    """
    Find MAP estimate
    """
    with model:
        mp = pm.find_MAP(include_transformed=True)

    return mp