"""
V1.0 - Modelos de conteo para el numero anual de grandes incendios (GIF).

Modelo principal previsto (Bayesiano):
    N_t ~ NegBin(mu_t, alpha)
    log(mu_t) = beta0 + x_t' beta + u_t
    u_t = rho u_{t-1} + eta_t

Contraste reproducible sin PyMC:
    NB2 con realimentacion del residuo de Pearson retardado, aproximacion
    observacional a GLARMA(1,0). La dispersion NB2 se estima por maxima
    verosimilitud y las predicciones simulan incertidumbre de parametros y
    de observacion.

El protocolo conserva el bloqueo temporal de análisis previo:
    - desarrollo: predicciones rodantes 2014-2023;
    - validacion externa: 2024-2025, sin usar sus respuestas para seleccionar.

Uso rapido (reproducible en este paquete):
    python modelos_conteo_gif_v13.py --input serie.csv --out salida --backend glarma

Uso bayesiano:
    python modelos_conteo_gif_v13.py --input serie.csv --out salida --backend pymc \
        --draws 1500 --tune 1500 --chains 4

Dependencias basicas: numpy, pandas, scipy, statsmodels, matplotlib.
Backend bayesiano: pymc y arviz compatibles entre si.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

warnings.filterwarnings("ignore")

PYMC_COMPAT_MODE = False
try:
    import pymc as pm
    HAS_PYMC = True
    PYMC_ERROR = ""
except Exception as first_exc:
    # Compatibilidad temporal para entornos con PyMC 5 y ArviZ 1.x.
    # Se usa MultiTrace (sin conversion a InferenceData), por lo que solo
    # hacen falta alias de importacion; no se modifican los calculos MCMC.
    try:
        import sys, types
        import arviz as _az
        import arviz_base as _ab
        from xarray import DataTree
        from collections.abc import Mapping as _Mapping
        _az.InferenceData = DataTree
        _az.concat = lambda *args, **kwargs: args[0][0] if args and args[0] else None
        _az.stats = types.SimpleNamespace(__all__=[])
        _az.plots = types.SimpleNamespace(__all__=[])
        _base = types.ModuleType("arviz.data.base")
        _base.CoordSpec = _Mapping; _base.DimSpec = _Mapping
        def _dict_to_dataset_compat(*args, **kwargs):
            kwargs.pop("library", None)
            return _ab.dict_to_dataset(*args, **kwargs)
        _base.dict_to_dataset = _dict_to_dataset_compat
        _base.make_attrs = _ab.make_attrs
        _base.requires = lambda attrs: (lambda fn: fn)
        _pkg = types.ModuleType("arviz.data"); _pkg.base = _base
        _inf = types.ModuleType("arviz.data.inference_data"); _inf.WARMUP_TAG = "warmup_"
        sys.modules["arviz.data"] = _pkg
        sys.modules["arviz.data.base"] = _base
        sys.modules["arviz.data.inference_data"] = _inf
        import pymc as pm
        HAS_PYMC = True
        PYMC_COMPAT_MODE = True
        PYMC_ERROR = f"Modo compatibilidad activado tras: {type(first_exc).__name__}: {first_exc}"
    except Exception as exc:
        HAS_PYMC = False
        PYMC_ERROR = f"{type(exc).__name__}: {exc}"

BASE_SPECS = {
    "NB-AR1": [],
    "NB-AR1+HE": ["heatwave_events"],
    "NB-AR1+P2": ["prevention_lag2"],
    "NB-AR1+I2": ["total_investment_lag2"],
    "NB-AR1+HE+P2": ["heatwave_events", "prevention_lag2"],
    "NB-AR1+HE+I2": ["heatwave_events", "total_investment_lag2"],
    "NB-AR1+HD+P2": ["heatwave_days", "prevention_lag2"],
}
MAIN_MODEL_ID = "NB-AR1+HE+P2"

NATURAL_EFFECTS = {
    "heatwave_days": (10.0, "+10 dias de ola de calor"),
    "heatwave_events": (1.0, "+1 episodio de ola de calor"),
    "prevention_lag2": (1.0, "+1 EUR real/ha de prevencion (t-2)"),
    "total_investment_lag2": (1.0, "+1 EUR real/ha de inversion total (t-2)"),
    "fwi_indicator": (1.0, "+1 unidad del indicador FWI"),
}


def stable_rng(*parts: object) -> np.random.Generator:
    text = "|".join(map(str, parts)).encode("utf-8")
    seed = int(hashlib.sha256(text).hexdigest()[:16], 16) % (2**32 - 1)
    return np.random.default_rng(seed)


def prepare_data(input_csv: Path, fwi_csv: Path | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    d = pd.read_csv(input_csv).sort_values("year").reset_index(drop=True)
    required = {
        "year", "count_gif", "heatwave_days", "heatwave_events",
        "prevention_eur_per_forest_ha", "total_investment_eur_per_forest_ha",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    d["prevention_lag2"] = d["prevention_eur_per_forest_ha"].shift(2)
    d["total_investment_lag2"] = d["total_investment_eur_per_forest_ha"].shift(2)
    specs = dict(BASE_SPECS)
    if fwi_csv and Path(fwi_csv).exists():
        fwi = pd.read_csv(fwi_csv)
        if "year" not in fwi:
            raise ValueError("El CSV FWI debe contener year.")
        numeric = [c for c in fwi.columns if c != "year" and pd.api.types.is_numeric_dtype(fwi[c])]
        if not numeric:
            raise ValueError("El CSV FWI debe contener una columna numerica adicional.")
        fcol = numeric[0]
        d = d.merge(fwi[["year", fcol]].rename(columns={fcol: "fwi_indicator"}), on="year", how="left")
        overlap = d.loc[d.year.between(2005, 2023), "fwi_indicator"].notna().sum()
        if overlap >= 10:
            specs["NB-AR1+FWI"] = ["fwi_indicator"]
            specs["NB-AR1+FWI+P2"] = ["fwi_indicator", "prevention_lag2"]
    return d, specs


def standardize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    if not cols:
        return None, None, {}, {}
    mu = train[cols].mean()
    sd = train[cols].std(ddof=0).replace(0, 1.0)
    xtr = ((train[cols] - mu) / sd).to_numpy(float)
    xte = ((test[cols] - mu) / sd).to_numpy(float)
    return xtr, xte, mu.to_dict(), sd.to_dict()


def _nb2_fit(y: np.ndarray, X: np.ndarray):
    return NegativeBinomial(y, X, loglike_method="nb2").fit(
        disp=False, maxiter=1000, method="bfgs"
    )


def fit_glarma_nb_model(y_train: np.ndarray, X_train: np.ndarray | None):
    """Aproximacion GLARMA(1,0)-NB2 con feedback del residuo de Pearson.

    Se estima primero una NB2 sin feedback. El residuo de Pearson retardado
    se introduce despues como regresor observacional. Es un contraste
    frecuentista parsimonioso; no se presenta como implementacion exacta del
    algoritmo R glarma.
    """
    T = len(y_train)
    X0 = np.ones((T, 1)) if X_train is None else np.column_stack([np.ones(T), X_train])
    base = _nb2_fit(y_train, X0)
    alpha0 = max(float(base.params[-1]), 1e-6)
    mu0 = np.clip(base.predict(), 1e-8, None)
    pearson = (y_train - mu0) / np.sqrt(mu0 + alpha0 * mu0**2)
    z = np.concatenate([[0.0], pearson[:-1]])
    Xg = np.column_stack([X0, z])
    final = _nb2_fit(y_train, Xg)
    return final, Xg


def simulate_glarma_forecast(
    fit, y_train: np.ndarray, X_test: np.ndarray | None,
    n_sim: int, rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Prediccion un paso con incertidumbre de parametros y observacion."""
    beta_hat = np.asarray(fit.params[:-1], float)
    alpha_hat = max(float(fit.params[-1]), 1e-6)
    covariance_ok = True
    try:
        cov = np.asarray(fit.cov_params(), float)
        cov_beta = cov[:-1, :-1]
    except Exception:
        # En pliegues muy cortos alpha puede quedar en el borde Poisson y
        # statsmodels no devuelve la covarianza completa. Se recupera la
        # curvatura de los coeficientes beta y se fija alpha en su MLE.
        covariance_ok = False
        h_beta = np.asarray(fit.model.hessian(fit.params), float)[:-1, :-1]
        cov_beta = np.linalg.pinv(-h_beta)
        cov = None

    mu_last = float(np.clip(fit.predict()[-1], 1e-8, None))
    z_next = (y_train[-1] - mu_last) / math.sqrt(mu_last + alpha_hat * mu_last**2)
    x_next = np.asarray([1.0] + ([] if X_test is None else list(np.ravel(X_test))) + [z_next], float)

    # Regulariza numericamente la covarianza por si la muestra corta produce
    # una matriz casi singular.
    cov_beta = (cov_beta + cov_beta.T) / 2
    eigval, eigvec = np.linalg.eigh(cov_beta)
    eigval = np.clip(eigval, 1e-10, None)
    cov_beta = eigvec @ np.diag(eigval) @ eigvec.T
    beta_draw = rng.multivariate_normal(beta_hat, cov_beta, size=n_sim)

    if covariance_ok and cov is not None and np.isfinite(cov[-1, -1]) and cov[-1, -1] > 0:
        se_alpha = float(np.sqrt(cov[-1, -1]))
        log_alpha_sd = min(se_alpha / alpha_hat, 2.0)
        alpha_draw = np.exp(rng.normal(np.log(alpha_hat), log_alpha_sd, size=n_sim))
    else:
        alpha_draw = np.full(n_sim, alpha_hat)
    eta = beta_draw @ x_next
    mu = np.exp(np.clip(eta, -20, 20))
    size = 1.0 / alpha_draw
    prob = size / (size + mu)
    sims = rng.negative_binomial(size, prob)
    point = float(np.exp(np.clip(beta_hat @ x_next, -20, 20)))
    return point, sims


def fit_predict_glarma_nb(y_train, X_train, X_test, n_sim=4000, rng=None):
    rng = rng or np.random.default_rng(20260728)
    fit, _ = fit_glarma_nb_model(np.asarray(y_train, int), X_train)
    return simulate_glarma_forecast(fit, np.asarray(y_train, int), X_test, n_sim, rng)


def sample_nb_ar1_model(y_train, X_train, draws=1500, tune=1500, chains=4,
                          random_seed=20260728):
    """Muestrea la NB con estado latente AR(1) y devuelve MultiTrace.

    Las covariables deben llegar estandarizadas dentro del conjunto de
    entrenamiento. El modelo usa la parametrizacion de PyMC en la que
    Var(N|mu,alpha)=mu+mu^2/alpha.
    """
    if not HAS_PYMC:
        raise RuntimeError(f"PyMC no disponible o incompatible: {PYMC_ERROR}")
    T = len(y_train)
    k = 0 if X_train is None else X_train.shape[1]
    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=np.log(max(np.mean(y_train), 1.0)), sigma=1.0)
        beta = pm.Normal("beta", 0.0, 1.0, shape=k) if k else None
        rho = pm.Uniform("rho", -0.98, 0.98)
        sigma_u = pm.HalfNormal("sigma_u", 0.5)
        # Parametrizacion no centrada: evita el embudo sigma_u--u en muestras
        # anuales cortas y estabiliza los pliegues de validacion.
        init_sd_raw = 1.0 / pm.math.sqrt(pm.math.maximum(1 - rho**2, 1e-4))
        u_raw = pm.AR(
            "u_raw", rho=rho, sigma=1.0, constant=False, shape=T,
            init_dist=pm.Normal.dist(0.0, init_sd_raw),
        )
        u = pm.Deterministic("u", sigma_u * u_raw)
        eta = beta0 + u
        if k:
            eta = eta + pm.math.dot(X_train, beta)
        alpha = pm.HalfNormal("alpha_disp", 10.0)
        pm.NegativeBinomial("N", mu=pm.math.exp(eta), alpha=alpha, observed=y_train)
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=min(chains, 2),
            target_accept=0.95, init="adapt_diag", random_seed=random_seed, progressbar=False,
            return_inferencedata=False, compute_convergence_checks=False,
        )
    return trace


def fit_predict_nb_ar1(y_train, X_train, X_test, draws=1500, tune=1500, chains=4, rng=None):
    rng = rng or np.random.default_rng(20260728)
    trace = sample_nb_ar1_model(y_train, X_train, draws, tune, chains)
    k = 0 if X_train is None else X_train.shape[1]
    b0 = trace.get_values("beta0", combine=True)
    rho_s = trace.get_values("rho", combine=True)
    sig_s = trace.get_values("sigma_u", combine=True)
    alp_s = trace.get_values("alpha_disp", combine=True)
    uT = trace.get_values("u", combine=True)[:, -1]
    u_next = rho_s * uT + rng.normal(0.0, sig_s)
    eta_next = b0 + u_next
    if k:
        b = trace.get_values("beta", combine=True).reshape(-1, k)
        eta_next += b @ np.ravel(X_test)
    mu_next = np.exp(np.clip(eta_next, -20, 20))
    p = alp_s / (alp_s + mu_next)
    return rng.negative_binomial(alp_s, p)


def _rhat_basic(chain_values) -> float:
    """R-hat clasico para una lista de cadenas; NaN si no es calculable."""
    arrays = [np.asarray(a, float).reshape(len(a), -1) for a in chain_values]
    if len(arrays) < 2:
        return float("nan")
    n = min(len(a) for a in arrays)
    if n < 4:
        return float("nan")
    # Se resume por componente y se devuelve el peor R-hat.
    arr = np.stack([a[:n] for a in arrays], axis=0)  # m,n,k
    means = arr.mean(axis=1)
    W = arr.var(axis=1, ddof=1).mean(axis=0)
    B = n * means.var(axis=0, ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(var_hat / W)
    return float(np.nanmax(r))


def full_sample_pymc_coefficients(d: pd.DataFrame, specs: dict[str, list[str]],
                                  model_ids: set[str], draws=1500, tune=1500,
                                  chains=4, end_year=2023):
    """Resumen posterior de los modelos bayesianos seleccionados."""
    rows = []
    diagnostics = []
    for model_id in model_ids:
        if model_id not in specs:
            continue
        cols = specs[model_id]
        needed = ["count_gif"] + cols
        train = d[d.year <= end_year].dropna(subset=needed)
        Xtr, _, _, sd = standardize(train, train.iloc[[0]], cols)
        trace = sample_nb_ar1_model(
            train.count_gif.to_numpy(int), Xtr, draws=draws, tune=tune,
            chains=chains, random_seed=20260728 + len(rows),
        )
        param_names = ["beta0", "rho", "sigma_u", "alpha_disp"]
        for name in param_names:
            vals = np.asarray(trace.get_values(name, combine=True), float).reshape(-1)
            rows.append({
                "model_id": model_id, "parameter": name,
                "posterior_mean": float(vals.mean()),
                "posterior_sd": float(vals.std(ddof=1)),
                "q025": float(np.quantile(vals, .025)),
                "q05": float(np.quantile(vals, .05)),
                "q50": float(np.quantile(vals, .5)),
                "q95": float(np.quantile(vals, .95)),
                "q975": float(np.quantile(vals, .975)),
                "prob_negative": float(np.mean(vals < 0)),
                "rhat_basic": _rhat_basic(trace.get_values(name, combine=False)),
            })
        if cols:
            b_all = np.asarray(trace.get_values("beta", combine=True), float).reshape(-1, len(cols))
            b_chains = trace.get_values("beta", combine=False)
            for j, col in enumerate(cols):
                vals = b_all[:, j]
                delta, unit = NATURAL_EFFECTS[col]
                irr = np.exp(vals * delta / sd[col])
                rows.append({
                    "model_id": model_id, "parameter": col,
                    "posterior_mean": float(vals.mean()),
                    "posterior_sd": float(vals.std(ddof=1)),
                    "q025": float(np.quantile(vals, .025)),
                    "q05": float(np.quantile(vals, .05)),
                    "q50": float(np.quantile(vals, .5)),
                    "q95": float(np.quantile(vals, .95)),
                    "q975": float(np.quantile(vals, .975)),
                    "prob_negative": float(np.mean(vals < 0)),
                    "rhat_basic": _rhat_basic([np.asarray(a)[:, j] for a in b_chains]),
                    "effect_unit": unit,
                    "IRR_mean_natural": float(irr.mean()),
                    "IRR_q025_natural": float(np.quantile(irr, .025)),
                    "IRR_q975_natural": float(np.quantile(irr, .975)),
                    "pct_change_mean_natural": float(100 * (irr.mean() - 1)),
                })
        diagnostics.append({
            "model_id": model_id, "n": len(train), "draws": draws,
            "tune": tune, "chains": chains,
            "max_rhat_basic": float(np.nanmax([
                r["rhat_basic"] for r in rows if r["model_id"] == model_id
            ])),
        })
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)

def crps_from_samples(samples: np.ndarray, obs: float) -> float:
    """CRPS empirico O(n log n), sin matriz n x n."""
    s = np.sort(np.asarray(samples, float))
    n = len(s)
    if n == 0:
        return float("nan")
    term1 = np.mean(np.abs(s - obs))
    weights = 2 * np.arange(1, n + 1) - n - 1
    half_pair_mean = np.sum(weights * s) / (n * n)
    return float(term1 - half_pair_mean)


def summarize_forecast(samples: np.ndarray, obs: float) -> dict:
    lo, hi = np.percentile(samples, [5, 95])
    mean = float(np.mean(samples))
    return {
        "predicted_mean": mean,
        "predicted_median": float(np.median(samples)),
        "pi90_low": float(lo),
        "pi90_high": float(hi),
        "covered_90": bool(lo <= obs <= hi),
        "crps": crps_from_samples(samples, obs),
        "abs_error": float(abs(mean - obs)),
    }


def rolling_validation(d, specs, backend="glarma", start=2014, end=2023,
                       n_sim=4000, draws=1500, tune=1500, chains=4,
                       pymc_models=None):
    rows = []
    for model_id, cols in specs.items():
        needed = ["count_gif"] + cols
        for year in range(start, end + 1):
            train = d[d.year < year].dropna(subset=needed)
            test = d[d.year == year].dropna(subset=needed)
            if test.empty or len(train) < 7:
                continue
            y = train.count_gif.to_numpy(int)
            obs = float(test.count_gif.iloc[0])
            Xtr, Xte, _, _ = standardize(train, test, cols)
            if backend in ("glarma", "both"):
                try:
                    rng = stable_rng("glarma", model_id, year)
                    _, sims = fit_predict_glarma_nb(y, Xtr, Xte, n_sim=n_sim, rng=rng)
                    rows.append({"backend": "NB2-feedback", "model_id": model_id,
                                 "year": year, "observed": obs,
                                 **summarize_forecast(sims, obs)})
                except Exception as exc:
                    rows.append({"backend": "NB2-feedback", "model_id": model_id,
                                 "year": year, "observed": obs, "error": str(exc)})
            if (backend in ("pymc", "both") and HAS_PYMC
                    and (pymc_models is None or model_id in pymc_models)):
                try:
                    rng = stable_rng("pymc", model_id, year)
                    sims = fit_predict_nb_ar1(y, Xtr, Xte, draws, tune, chains, rng)
                    rows.append({"backend": "NB-AR1 (PyMC)", "model_id": model_id,
                                 "year": year, "observed": obs,
                                 **summarize_forecast(sims, obs)})
                except Exception as exc:
                    rows.append({"backend": "NB-AR1 (PyMC)", "model_id": model_id,
                                 "year": year, "observed": obs, "error": str(exc)})
    return pd.DataFrame(rows)


def metrics_table(pred: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    dev = d[d.year.between(2007, 2023)].count_gif.dropna().to_numpy(float)
    naive = float(np.mean(np.abs(np.diff(dev))))
    rows = []
    good = pred.dropna(subset=["predicted_mean"]) if len(pred) else pred
    for (backend, model_id), g in good.groupby(["backend", "model_id"]):
        if len(g) < 8:
            continue
        y = g.observed.to_numpy(float)
        p = g.predicted_mean.to_numpy(float)
        mae = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append({
            "backend": backend, "model_id": model_id, "n": len(g),
            "MAE": mae, "RMSE": rmse, "NMAE": mae / y.mean(),
            "NRMSE": rmse / y.mean(), "MASE": mae / naive,
            "R2_pred": 1 - np.sum((y - p) ** 2) / ss_tot,
            "CRPS_mean": float(g.crps.mean()),
            "coverage_90": float(g.covered_90.mean()),
            "mean_PI90_width": float((g.pi90_high - g.pi90_low).mean()),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["Skill_MSE_vs_endogenous"] = np.nan
    out["Skill_MAE_vs_endogenous"] = np.nan
    for backend, g in out.groupby("backend"):
        ref = g[g.model_id == "NB-AR1"]
        if ref.empty:
            continue
        ref_rmse = float(ref.RMSE.iloc[0])
        ref_mae = float(ref.MAE.iloc[0])
        idx = g.index
        out.loc[idx, "Skill_MSE_vs_endogenous"] = 1 - (out.loc[idx, "RMSE"] ** 2) / (ref_rmse ** 2)
        out.loc[idx, "Skill_MAE_vs_endogenous"] = 1 - out.loc[idx, "MAE"] / ref_mae
    return out.sort_values(["backend", "CRPS_mean", "MAE"]).reset_index(drop=True)


def external_validation(d, specs, backend="glarma", n_sim=10000,
                        draws=2000, tune=2000, chains=4, pymc_models=None):
    rows = []
    for model_id, cols in specs.items():
        needed = ["count_gif"] + cols
        train = d[d.year <= 2023].dropna(subset=needed)
        for year in (2024, 2025):
            test = d[d.year == year].dropna(subset=needed)
            if test.empty:
                continue
            y = train.count_gif.to_numpy(int)
            obs = float(test.count_gif.iloc[0])
            Xtr, Xte, _, _ = standardize(train, test, cols)
            if backend in ("glarma", "both"):
                rng = stable_rng("external-glarma", model_id, year)
                _, sims = fit_predict_glarma_nb(y, Xtr, Xte, n_sim=n_sim, rng=rng)
                rows.append({"backend": "NB2-feedback", "model_id": model_id,
                             "year": year, "observed": obs,
                             **summarize_forecast(sims, obs)})
            if (backend in ("pymc", "both") and HAS_PYMC
                    and (pymc_models is None or model_id in pymc_models)):
                rng = stable_rng("external-pymc", model_id, year)
                sims = fit_predict_nb_ar1(y, Xtr, Xte, draws, tune, chains, rng)
                rows.append({"backend": "NB-AR1 (PyMC)", "model_id": model_id,
                             "year": year, "observed": obs,
                             **summarize_forecast(sims, obs)})
    return pd.DataFrame(rows)


def full_sample_coefficients(d: pd.DataFrame, specs: dict[str, list[str]], end_year=2023):
    rows = []
    diagnostics = []
    for model_id, cols in specs.items():
        needed = ["count_gif"] + cols
        train = d[d.year <= end_year].dropna(subset=needed)
        if len(train) < 8:
            continue
        Xtr, _, mu, sd = standardize(train, train.iloc[[0]], cols)
        fit, _ = fit_glarma_nb_model(train.count_gif.to_numpy(int), Xtr)
        names = ["intercept"] + cols + ["feedback_ar1", "alpha_nb2"]
        params = np.asarray(fit.params, float)
        bse = np.asarray(fit.bse, float)
        pval = np.asarray(fit.pvalues, float)
        ci = np.asarray(fit.conf_int(), float)
        for j, name in enumerate(names):
            row = {
                "model_id": model_id, "parameter": name, "estimate": params[j],
                "std_error": bse[j], "p_value": pval[j],
                "ci95_low": ci[j, 0], "ci95_high": ci[j, 1],
                "standardized": name in cols,
            }
            if name in cols:
                delta, unit = NATURAL_EFFECTS[name]
                scale = sd[name]
                row.update({
                    "effect_unit": unit,
                    "IRR_natural": float(np.exp(params[j] * delta / scale)),
                    "IRR_ci95_low": float(np.exp(ci[j, 0] * delta / scale)),
                    "IRR_ci95_high": float(np.exp(ci[j, 1] * delta / scale)),
                    "pct_change_natural": float(100 * np.expm1(params[j] * delta / scale)),
                })
            rows.append(row)
        diagnostics.append({
            "model_id": model_id, "n": len(train), "loglike": float(fit.llf),
            "AIC": float(fit.aic), "BIC": float(fit.bic),
            "alpha_nb2": float(params[-1]), "converged": bool(fit.mle_retvals.get("converged", True)),
        })
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def overdispersion_summary(d: pd.DataFrame, end_year=2023) -> pd.DataFrame:
    y = d.loc[d.year.between(2005, end_year), "count_gif"].dropna().to_numpy(float)
    return pd.DataFrame([{
        "n": len(y), "mean_count": y.mean(), "variance_count": y.var(ddof=1),
        "variance_to_mean": y.var(ddof=1) / y.mean(),
        "min_count": y.min(), "max_count": y.max(),
    }])


def make_figures(pred: pd.DataFrame, metrics: pd.DataFrame, ext: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    if len(metrics):
        g = metrics[metrics.backend == "NB2-feedback"].sort_values("CRPS_mean")
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.barh(g.model_id, g.CRPS_mean)
        ax.invert_yaxis(); ax.set_xlabel("CRPS medio (menor es mejor)")
        ax.set_title("Validacion rodante 2014-2023: modelos de conteo")
        fig.tight_layout(); fig.savefig(out / "fig_crps_modelos.pdf"); fig.savefig(out / "fig_crps_modelos.png", dpi=200)
        plt.close(fig)
    good = pred[(pred.backend == "NB2-feedback") & pred.model_id.isin(["NB-AR1", MAIN_MODEL_ID])]
    if len(good):
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        obs = good.drop_duplicates("year").sort_values("year")
        ax.plot(obs.year, obs.observed, marker="o", linewidth=2, label="Observado")
        for mid, grp in good.groupby("model_id"):
            grp = grp.sort_values("year")
            label = "NB endogeno" if mid == "NB-AR1" else "NB calor+prevencion"
            ax.plot(grp.year, grp.predicted_mean, marker="o", label=label)
            ax.fill_between(grp.year, grp.pi90_low, grp.pi90_high, alpha=0.15)
        ax.set_ylabel("Numero de GIF"); ax.set_xlabel("Ano")
        ax.set_title("Prediccion rodante e intervalos predictivos del 90%")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "fig_rolling_conteo_pi90.pdf"); fig.savefig(out / "fig_rolling_conteo_pi90.png", dpi=200)
        plt.close(fig)
    eg = ext[(ext.backend == "NB2-feedback") & ext.model_id.isin(["NB-AR1", MAIN_MODEL_ID])]
    if len(eg):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        x = np.arange(len(eg))
        yerr = np.vstack([eg.predicted_mean - eg.pi90_low, eg.pi90_high - eg.predicted_mean])
        ax.errorbar(x, eg.predicted_mean, yerr=yerr, fmt="o", capsize=4, label="Prediccion e IP90")
        ax.scatter(x, eg.observed, marker="x", s=70, label="Observado")
        ax.set_xticks(x, [f"{r.model_id}\n{int(r.year)}" for _, r in eg.iterrows()], rotation=25, ha="right")
        ax.set_ylabel("Numero de GIF"); ax.set_title("Validacion externa bloqueada")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "fig_external_conteo_pi90.pdf"); fig.savefig(out / "fig_external_conteo_pi90.png", dpi=200)
        plt.close(fig)


FWI_ERA5_NOTE = """\
Fuente observacional recomendada:
Fire danger indices historical data from the Copernicus Emergency Management Service
(dataset cems-fire-historical-v1; DOI 10.24381/cds.0e89c522).

Descargar FWI diario, version 4.1, tipo consolidated, 2005-2023. Agregar con
mascara de Espana/CCAA y ponderacion por cos(lat). Indicador principal
pre-registrado: dias equivalentes con FWI >= 38 entre mayo y octubre. No usar
las proyecciones RCP para completar el pasado.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fwi-csv", type=Path, default=None)
    ap.add_argument("--backend", choices=["pymc", "glarma", "both"], default="glarma")
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--n-sim", type=int, default=6000)
    ap.add_argument("--pymc-models", default="NB-AR1,NB-AR1+HE+P2",
                    help="Modelos bayesianos separados por coma; limita el coste MCMC.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend in ("pymc", "both") and not HAS_PYMC:
        print("AVISO: backend PyMC no ejecutado:", PYMC_ERROR)
        if args.backend == "pymc":
            raise SystemExit(2)
    d, specs = prepare_data(args.input, args.fwi_csv)
    pymc_models = {x.strip() for x in args.pymc_models.split(",") if x.strip()}
    pred = rolling_validation(d, specs, args.backend, n_sim=args.n_sim,
                              draws=args.draws, tune=args.tune, chains=args.chains,
                              pymc_models=pymc_models)
    met = metrics_table(pred, d)
    ext = external_validation(d, specs, args.backend, n_sim=args.n_sim,
                              draws=args.draws, tune=args.tune,
                              chains=args.chains, pymc_models=pymc_models)
    coef, diag = full_sample_coefficients(d, specs)
    over = overdispersion_summary(d)
    post, post_diag = (pd.DataFrame(), pd.DataFrame())
    if args.backend in ("pymc", "both") and HAS_PYMC:
        post, post_diag = full_sample_pymc_coefficients(
            d, specs, pymc_models, draws=args.draws, tune=args.tune,
            chains=args.chains
        )

    pred.to_csv(args.out / "predicciones_rodantes_conteo.csv", index=False)
    met.to_csv(args.out / "metricas_rodantes_conteo.csv", index=False)
    ext.to_csv(args.out / "validacion_externa_conteo.csv", index=False)
    coef.to_csv(args.out / "coeficientes_nb2_feedback.csv", index=False)
    diag.to_csv(args.out / "diagnosticos_nb2_feedback.csv", index=False)
    over.to_csv(args.out / "sobredispersion_conteos.csv", index=False)
    if len(post):
        post.to_csv(args.out / "coeficientes_posteriores_nb_ar1.csv", index=False)
        post_diag.to_csv(args.out / "diagnosticos_posteriores_nb_ar1.csv", index=False)
    make_figures(pred, met, ext, args.out / "figuras")
    (args.out / "NOTA_FWI.txt").write_text(FWI_ERA5_NOTE, encoding="utf-8")
    status = {
        "backend_requested": args.backend,
        "pymc_available": HAS_PYMC,
        "pymc_import_error": PYMC_ERROR,
        "pymc_compat_mode": PYMC_COMPAT_MODE,
        "pymc_models": sorted(pymc_models),
        "python": platform.python_version(),
        "models": list(specs),
        "fwi_models_activated": any("FWI" in x for x in specs),
    }
    (args.out / "estado_ejecucion.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(met.to_string(index=False))
    print("\nValidacion externa:")
    keep = ["backend", "model_id", "year", "observed", "predicted_mean", "pi90_low", "pi90_high", "covered_90", "crps"]
    print(ext[keep].to_string(index=False))


if __name__ == "__main__":
    main()
