from __future__ import annotations

import argparse
import ast
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

ARIMA_ORDERS = [(p, d, q) for d in (0, 1) for p in range(4) for q in range(4) if (p + d + q > 0) and not (d == 1 and p + q == 6)]
ARIMAX_ORDERS = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]


def prepare_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    for lag in (1, 2, 3):
        d[f"prevention_lag{lag}"] = d["prevention_eur_per_forest_ha"].shift(lag)
        d[f"investment_lag{lag}"] = d["total_investment_eur_per_forest_ha"].shift(lag)
    d["log_count1p"] = np.log1p(d["count_gif"].astype(float))
    return d


def inv_log_median(mu: float) -> float:
    return max(0.0, float(np.expm1(min(15.0, mu))))


def metrics(y: np.ndarray, pred: np.ndarray, scale: float) -> dict[str, float]:
    mae = float(np.mean(np.abs(y - pred)))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "NMAE": mae / float(np.mean(y)),
        "NRMSE": rmse / float(np.mean(y)),
        "MASE": mae / scale,
        "R2_pred": float(r2_score(y, pred)),
    }


def fit_sarimax(y: np.ndarray, x: np.ndarray | None, order: tuple[int, int, int]):
    trend = "c" if order[1] == 0 else "t"
    return SARIMAX(
        y,
        exog=x,
        order=order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=250)


def rolling_arima(d: pd.DataFrame, start: int = 2014, end: int = 2023):
    scale = float(np.mean(np.abs(np.diff(d.loc[d.year.between(2005, 2023), "count_gif"]))))
    rows, preds = [], []
    for order in ARIMA_ORDERS:
        fold = []
        for year in range(start, end + 1):
            train = d[d.year < year].reset_index(drop=True)
            test = d[d.year == year]
            k = 2 + order[0] + order[2]
            if len(train) < max(8, k + 3):
                continue
            try:
                res = fit_sarimax(train.log_count1p.to_numpy(), None, order)
                pred = inv_log_median(float(res.forecast(1)[0]))
                if np.isfinite(pred) and pred < 1e5:
                    fold.append((year, float(test.count_gif.iloc[0]), pred))
            except Exception:
                continue
        if len(fold) >= 8:
            y = np.array([r[1] for r in fold])
            p = np.array([r[2] for r in fold])
            rows.append({"family": "ARIMA", "spec": "endogenous", "order": str(order), "n": len(fold), **metrics(y, p, scale)})
            for year, obs, pred in fold:
                preds.append({"family": "ARIMA", "spec": "endogenous", "order": str(order), "year": year, "observed": obs, "predicted": pred})

    for name in ("naive", "mean", "SES", "Holt", "Holt_damped", "AutoReg1", "AutoReg2", "AutoReg3"):
        fold = []
        for year in range(start, end + 1):
            train = d[d.year < year]
            test = d[d.year == year]
            y = train.count_gif.astype(float)
            try:
                if name == "naive":
                    pred = float(y.iloc[-1])
                elif name == "mean":
                    pred = float(y.mean())
                elif name == "SES":
                    pred = float(SimpleExpSmoothing(y, initialization_method="estimated").fit().forecast(1).iloc[0])
                elif name == "Holt":
                    pred = float(Holt(y, initialization_method="estimated").fit().forecast(1).iloc[0])
                elif name == "Holt_damped":
                    pred = float(Holt(y, damped_trend=True, initialization_method="estimated").fit().forecast(1).iloc[0])
                else:
                    lag = int(name[-1])
                    pred = float(np.asarray(AutoReg(y.to_numpy(), lags=lag, trend="c", old_names=False).fit().predict(len(y), len(y)))[0])
                fold.append((year, float(test.count_gif.iloc[0]), max(0.0, pred)))
            except Exception:
                continue
        if len(fold) >= 8:
            yy = np.array([r[1] for r in fold])
            pp = np.array([r[2] for r in fold])
            rows.append({"family": "baseline", "spec": name, "order": "-", "n": len(fold), **metrics(yy, pp, scale)})
            for year, obs, pred in fold:
                preds.append({"family": "baseline", "spec": name, "order": "-", "year": year, "observed": obs, "predicted": pred})
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]), pd.DataFrame(preds)


def arimax_specs() -> dict[str, list[str]]:
    specs = {"heat_days": ["heatwave_days"], "heat_events": ["heatwave_events"]}
    for lag in (1, 2, 3):
        specs[f"heat_days+prev_lag{lag}"] = ["heatwave_days", f"prevention_lag{lag}"]
        specs[f"heat_events+prev_lag{lag}"] = ["heatwave_events", f"prevention_lag{lag}"]
        specs[f"heat_days+invest_lag{lag}"] = ["heatwave_days", f"investment_lag{lag}"]
        specs[f"heat_events+invest_lag{lag}"] = ["heatwave_events", f"investment_lag{lag}"]
    return specs


def rolling_arimax(d: pd.DataFrame, start: int = 2014, end: int = 2023):
    scale = float(np.mean(np.abs(np.diff(d.loc[d.year.between(2005, 2023), "count_gif"]))))
    rows, preds = [], []
    for spec, cols in arimax_specs().items():
        for order in ARIMAX_ORDERS:
            fold = []
            for year in range(start, end + 1):
                train = d[d.year < year].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
                test = d[d.year == year].dropna(subset=cols).reset_index(drop=True)
                if test.empty or len(train) < max(8, 3 + len(cols) + order[0] + order[2]):
                    continue
                try:
                    scaler = StandardScaler().fit(train[cols])
                    x_train = scaler.transform(train[cols])
                    x_test = scaler.transform(test[cols])
                    res = fit_sarimax(train.log_count1p.to_numpy(), x_train, order)
                    pred = inv_log_median(float(res.get_forecast(1, exog=x_test).predicted_mean[0]))
                    if np.isfinite(pred) and pred < 1e5:
                        fold.append((year, float(test.count_gif.iloc[0]), pred))
                except Exception:
                    continue
            if len(fold) >= 8:
                y = np.array([r[1] for r in fold])
                p = np.array([r[2] for r in fold])
                rows.append({"family": "ARIMAX", "spec": spec, "order": str(order), "n": len(fold), **metrics(y, p, scale)})
                for year, obs, pred in fold:
                    preds.append({"family": "ARIMAX", "spec": spec, "order": str(order), "year": year, "observed": obs, "predicted": pred})
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]), pd.DataFrame(preds)


def aicc_grid(d: pd.DataFrame) -> pd.DataFrame:
    train = d[d.year <= 2023].reset_index(drop=True)
    rows = []
    for order in ARIMA_ORDERS:
        try:
            res = fit_sarimax(train.log_count1p.to_numpy(), None, order)
            n, k = len(train), len(res.params)
            aicc = float(res.aic + 2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else np.nan
            rows.append({"order": str(order), "AIC": float(res.aic), "AICc": aicc, "BIC": float(res.bic), "converged": bool(res.mle_retvals.get("converged", True))})
        except Exception:
            rows.append({"order": str(order), "AIC": np.nan, "AICc": np.nan, "BIC": np.nan, "converged": False})
    return pd.DataFrame(rows)


def selected_classical_models(d: pd.DataFrame):
    specs = {
        "ARIMA(1,0,0)": ([], (1, 0, 0)),
        "ARIMAX calor+prevención t-1": (["heatwave_days", "prevention_lag1"], (0, 0, 0)),
        "ARIMAX calor+prevención t-2": (["heatwave_days", "prevention_lag2"], (0, 0, 0)),
        "ARIMAX calor+prevención t-3": (["heatwave_days", "prevention_lag3"], (0, 0, 0)),
        "ARIMAX episodios+inversión t-2": (["heatwave_events", "investment_lag2"], (0, 0, 0)),
    }
    coef_rows, external_rows = [], []
    for name, (cols, order) in specs.items():
        train = d[d.year <= 2023].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
        res = fit_sarimax(train.log_count1p.to_numpy(), train[cols].to_numpy() if cols else None, order)
        n, k = len(train), len(res.params)
        aicc = float(res.aic + 2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else np.nan
        names = list(res.param_names)
        for term, b, se, p, (lo, hi) in zip(names, res.params, res.bse, res.pvalues, res.conf_int()):
            coef_rows.append({"model": name, "term": term, "coefficient": float(b), "std_error": float(se), "p_value": float(p), "ci95_low": float(lo), "ci95_high": float(hi), "AIC": float(res.aic), "AICc": aicc, "BIC": float(res.bic), "n": n})

        # Sequential one-step external validation: update the state with 2024 before forecasting 2025.
        for year in (2024, 2025):
            tr = d[d.year < year].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
            te = d[d.year == year].dropna(subset=cols).reset_index(drop=True)
            if te.empty:
                continue
            rr = fit_sarimax(tr.log_count1p.to_numpy(), tr[cols].to_numpy() if cols else None, order)
            fc = rr.get_forecast(1, exog=te[cols].to_numpy() if cols else None)
            mu = float(fc.predicted_mean[0])
            se = float(fc.se_mean[0])
            observed = float(te.count_gif.iloc[0])
            obs_log = math.log1p(observed)
            low = max(0.0, float(np.expm1(mu + norm.ppf(0.05) * se)))
            high = max(0.0, float(np.expm1(mu + norm.ppf(0.95) * se)))
            z = (obs_log - mu) / se
            external_rows.append({
                "model": name,
                "year": year,
                "observed": observed,
                "predicted": inv_log_median(mu),
                "pi90_low": low,
                "pi90_high": high,
                "standardized_log_residual": float(z),
                "one_sided_upper_tail_p": float(1 - norm.cdf(z)),
            })
    return pd.DataFrame(coef_rows), pd.DataFrame(external_rows)


def common_lag_sensitivity(d: pd.DataFrame, start: int = 2016, end: int = 2023) -> pd.DataFrame:
    """Compare heat/prevention lags on exactly the same validation years.

    R2_pred is the out-of-sample coefficient 1-SSE/SST on the common validation
    window. It can be negative. Skill_MSE_ARIMA100 uses ARIMA(1,0,0) as the
    operational benchmark and is the preferred comparative statistic.
    """
    scale = float(np.mean(np.abs(np.diff(d.loc[d.year.between(2005, 2023), "count_gif"]))))
    specs = [
        ("calor+prevención t-1", ["heatwave_days", "prevention_lag1"], (0, 0, 0)),
        ("calor+prevención t-2", ["heatwave_days", "prevention_lag2"], (0, 0, 0)),
        ("calor+prevención t-3", ["heatwave_days", "prevention_lag3"], (0, 0, 0)),
        ("calor solo", ["heatwave_days"], (0, 0, 0)),
        ("ARIMA(1,0,0)", [], (1, 0, 0)),
    ]
    pred_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for name, cols, order in specs:
        fold = []
        for year in range(start, end + 1):
            tr = d[d.year < year].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
            te = d[d.year == year].dropna(subset=cols).reset_index(drop=True)
            res = fit_sarimax(tr.log_count1p.to_numpy(), tr[cols].to_numpy() if cols else None, order)
            fc = res.get_forecast(1, exog=te[cols].to_numpy() if cols else None)
            fold.append((float(te.count_gif.iloc[0]), inv_log_median(float(fc.predicted_mean[0]))))
        yy, pp = np.array([x[0] for x in fold]), np.array([x[1] for x in fold])
        pred_store[name] = (yy, pp)
        row = {"spec": name, "order": str(order), "n": len(fold), **metrics(yy, pp, scale)}
        if cols:
            full = d[d.year <= 2023].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
            rr = fit_sarimax(full.log_count1p.to_numpy(), full[cols].to_numpy(), order)
            row["heat_coefficient"] = float(rr.params[1])
            row["heat_p_value"] = float(rr.pvalues[1])
            if len(cols) == 2:
                row["prevention_coefficient"] = float(rr.params[2])
                row["prevention_p_value"] = float(rr.pvalues[2])
            # For order (0,0,0), compute the ordinary log-scale fit directly.
            if order == (0, 0, 0):
                eta = float(rr.params[0]) + full[cols].to_numpy() @ np.asarray(rr.params[1:1+len(cols)], dtype=float)
                r2_fit = float(r2_score(full.log_count1p.to_numpy(), eta))
                n, k = len(full), len(cols)
                row["R2_fit_log"] = r2_fit
                row["R2_fit_log_adjusted"] = 1 - (1-r2_fit) * (n-1) / (n-k-1)
        rows.append(row)
    bench_y, bench_p = pred_store["ARIMA(1,0,0)"]
    bench_sse = float(np.sum((bench_y - bench_p) ** 2))
    for row in rows:
        yy, pp = pred_store[row["spec"]]
        row["Skill_MSE_vs_ARIMA100"] = 1 - float(np.sum((yy-pp)**2)) / bench_sse
        row["RMSE_reduction_vs_ARIMA100_pct"] = 100 * (1 - row["RMSE"] / float(np.sqrt(bench_sse / len(bench_y))))
    return pd.DataFrame(rows)


def fitted_and_external_for_figure(d: pd.DataFrame) -> pd.DataFrame:
    """Create in-sample fitted trajectories (2005-2023) plus sequential 2024-2025 forecasts.

    The figure is descriptive: the 2005-2023 section is an in-sample fit, while
    2024-2025 are genuine sequential one-step forecasts. Contemporaneous heat
    makes the ARIMAX forecast conditional on the year's observed/forecast heat.
    """
    models = {
        "ARIMA(1,0,0)": ([], (1, 0, 0)),
        "ARIMAX calor + prevención t-1": (["heatwave_days", "prevention_lag1"], (0, 0, 0)),
    }
    rows = []
    for name, (cols, order) in models.items():
        train = d[d.year <= 2023].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
        x_train = train[cols].to_numpy() if cols else None
        res = fit_sarimax(train.log_count1p.to_numpy(), x_train, order)
        if order == (0, 0, 0) and cols:
            # Direct regression predictor avoids the diffuse-state artefact in the first fitted value.
            fitted = float(res.params[0]) + x_train @ np.asarray(res.params[1:1+len(cols)], dtype=float)
        else:
            fitted = np.asarray(res.fittedvalues, dtype=float)
            if len(fitted):
                fitted[0] = np.nan  # do not plot the diffuse initial-state placeholder
        for year, mu in zip(train.year, fitted):
            if np.isfinite(mu):
                rows.append({"model": name, "year": int(year), "predicted": inv_log_median(float(mu)), "phase": "fit", "pi90_low": np.nan, "pi90_high": np.nan})

        for year in (2024, 2025):
            tr = d[d.year < year].dropna(subset=["log_count1p"] + cols).reset_index(drop=True)
            te = d[d.year == year].dropna(subset=cols).reset_index(drop=True)
            rr = fit_sarimax(tr.log_count1p.to_numpy(), tr[cols].to_numpy() if cols else None, order)
            fc = rr.get_forecast(1, exog=te[cols].to_numpy() if cols else None)
            mu, se = float(fc.predicted_mean[0]), float(fc.se_mean[0])
            rows.append({
                "model": name, "year": year, "predicted": inv_log_median(mu), "phase": "forecast",
                "pi90_low": max(0.0, float(np.expm1(mu + norm.ppf(0.05) * se))),
                "pi90_high": max(0.0, float(np.expm1(mu + norm.ppf(0.95) * se))),
            })
    return pd.DataFrame(rows)

def plot_series(d: pd.DataFrame, preds: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10.4, 5.7), dpi=180)
    ax.axvspan(2023.5, 2025.5, color="0.90", zorder=0)
    ax.plot(d.year, d.count_gif, marker="o", linewidth=2.4, label="Observado", zorder=4)
    styles = {"ARIMA(1,0,0)": ("--", "s"), "ARIMAX calor + prevención t-1": ("-.", "D")}
    for name, (ls, marker) in styles.items():
        z = preds[preds.model == name].sort_values("year")
        ax.plot(z.year, z.predicted, linestyle=ls, marker=marker, linewidth=1.9, markersize=4.8, label=name, zorder=5)
        f = z[z.phase == "forecast"]
        ax.fill_between(f.year.to_numpy(float), f.pi90_low.to_numpy(float), f.pi90_high.to_numpy(float), alpha=0.16, zorder=1)
    p25 = preds[(preds.model == "ARIMAX calor + prevención t-1") & (preds.year == 2025)].iloc[0]
    observed25 = float(d.loc[d.year == 2025, "count_gif"].iloc[0])
    ax.annotate(
        f"2025: {int(observed25)} observados\n{p25.predicted:.1f} predichos; IP90% {p25.pi90_low:.0f}-{p25.pi90_high:.0f}",
        xy=(2025, observed25), xytext=(2021.2, 75),
        arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=8.5,
    )
    ax.axvline(2023.5, color="0.55", linewidth=1)
    ax.text(2014.3, 88, "Ajuste 2005-2023", ha="center", fontsize=8.8)
    ax.text(2024.35, 88, "Validación externa", ha="center", fontsize=8.8)
    ax.set(title="Evolución observada, ajuste y predicción externa", xlabel="Año", ylabel="Número de GIF", xlim=(2005, 2025), ylim=(0, 98))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.4, frameon=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

def plot_arima_heatmap(metric_table: pd.DataFrame, out: Path):
    z = metric_table[metric_table.family == "ARIMA"].copy()
    z[["p", "d", "q"]] = z.order.apply(lambda s: pd.Series(ast.literal_eval(s)))
    matrix = np.full((8, 4), np.nan)
    labels = []
    for dval in (0, 1):
        for pval in range(4):
            row = dval * 4 + pval
            labels.append(f"d={dval}, p={pval}")
            for qval in range(4):
                a = z[(z.p == pval) & (z.d == dval) & (z.q == qval)]
                if not a.empty:
                    matrix[row, qval] = float(a.MAE.iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 5.1), dpi=180)
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(4), [f"q={q}" for q in range(4)])
    ax.set_yticks(range(8), labels)
    for i in range(8):
        for j in range(4):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    finite = np.argwhere(np.isfinite(matrix))
    if len(finite):
        i, j = finite[np.argmin([matrix[tuple(x)] for x in finite])]
        ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, linewidth=2.2))
    ax.set_title("MAE rodante de las 30 especificaciones ARIMA")
    ax.set_xlabel("Orden de media móvil")
    ax.set_ylabel("Diferenciación y orden autorregresivo")
    fig.colorbar(im, ax=ax, label="MAE (GIF)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_2025_lag_sensitivity(external: pd.DataFrame, out: Path):
    names = [
        "ARIMA(1,0,0)",
        "ARIMAX calor+prevención t-3",
        "ARIMAX calor+prevención t-2",
        "ARIMAX calor+prevención t-1",
    ]
    z = external[(external.year == 2025) & (external.model.isin(names))].set_index("model").loc[names].reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=180)
    y = np.arange(len(z))
    x = z.predicted.to_numpy(float)
    lo = x - z.pi90_low.to_numpy(float)
    hi = z.pi90_high.to_numpy(float) - x
    ax.errorbar(x, y, xerr=[lo, hi], fmt="o", capsize=4, linewidth=1.7)
    ax.axvline(63, linestyle="--", linewidth=1.4, label="Observado: 63")
    ax.set_yticks(y, ["ARIMA(1,0,0)", "ARIMAX t-3", "ARIMAX t-2", "ARIMAX t-1"])
    ax.set_xlabel("Predicción de GIF en 2025 e intervalo predictivo del 90%")
    ax.set_title("Sensibilidad del nowcast de 2025 al retardo de prevención")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_posterior_effects(out: Path):
    labels = ["Calor: +10 días", "Prevención: +1 €/ha (t-2)"]
    irr = np.array([1.587, 1.095])
    low = np.array([1.169, 0.966])
    high = np.array([2.081, 1.227])
    probs = ["P(β > 0) = 0,997", "P(β < 0) = 0,070"]
    y = np.array([0.65, 0.35])
    fig, ax = plt.subplots(figsize=(8.4, 2.65), dpi=180)
    ax.errorbar(irr, y, xerr=[irr - low, high - irr], fmt="o", capsize=4, linewidth=1.8, markersize=6)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.88, 2.35)
    ax.set_ylim(0.18, 0.82)
    ax.set_xticks([0.9, 1.0, 1.2, 1.5, 2.0], ["0,9", "1,0", "1,2", "1,5", "2,0"])
    ax.set_yticks(y, labels)
    ax.set_xlabel("Razón de incidencia posterior (IRR)")
    ax.set_title("Efectos posteriores e intervalos creíbles del 95%")
    for xi, yi, text in zip(high, y, probs):
        ax.text(xi * 1.025, yi, text, va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("datos/serie_modelizacion_2005_2025.csv"))
    parser.add_argument("--output", type=Path, default=Path("resultados_v13_clasicos"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figuras").mkdir(exist_ok=True)
    d = prepare_data(args.input)
    arima_metrics, arima_preds = rolling_arima(d)
    arimax_metrics, arimax_preds = rolling_arimax(d)
    aicc = aicc_grid(d)
    coefs, external = selected_classical_models(d)
    common_lags = common_lag_sensitivity(d)
    selected = fitted_and_external_for_figure(d)
    arima_metrics.to_csv(args.output / "metricas_grid_arima_y_baselines.csv", index=False)
    arima_preds.to_csv(args.output / "predicciones_grid_arima.csv", index=False)
    arimax_metrics.to_csv(args.output / "metricas_sensibilidad_arimax.csv", index=False)
    arimax_preds.to_csv(args.output / "predicciones_sensibilidad_arimax.csv", index=False)
    aicc.to_csv(args.output / "aicc_grid_arima.csv", index=False)
    coefs.to_csv(args.output / "coeficientes_modelos_clasicos.csv", index=False)
    external.to_csv(args.output / "validacion_externa_secuencial.csv", index=False)
    common_lags.to_csv(args.output / "sensibilidad_retardos_comun_2016_2023.csv", index=False)
    selected.to_csv(args.output / "predicciones_ajuste_y_externo_figura.csv", index=False)
    adf = adfuller(d.loc[d.year <= 2023, "log_count1p"], autolag="AIC")
    pd.DataFrame([{"series": "log1p_count_gif", "ADF_statistic": adf[0], "p_value": adf[1]}]).to_csv(args.output / "adf_respuesta.csv", index=False)
    plot_series(d, selected, args.output / "figuras" / "fig_evolucion_arima_arimax")
    plot_arima_heatmap(arima_metrics, args.output / "figuras" / "fig_grid_arima_mae")
    plot_2025_lag_sensitivity(external, args.output / "figuras" / "fig_2025_sensibilidad_retardos")
    plot_posterior_effects(args.output / "figuras" / "fig_efectos_posteriores_compacta")


if __name__ == "__main__":
    main()
