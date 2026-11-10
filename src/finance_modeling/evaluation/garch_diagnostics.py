from __future__ import annotations

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import acf

from ..utils import logger


LJUNG_BOX_LAGS = (10, 20, 96)
ARCH_LM_LAGS = (10, 20)
ACF_LAGS = 192
DISTRIBUTION_PARAMS = {"nu", "eta", "lambda"}

SIGNIFICANCE = 0.05
MAX_PERSISTENCE = 0.999
RESIDUAL_RULES = {
    "ljung_box_z_p20": "residual autocorrelation",
    "ljung_box_z2_p20": "squared residual autocorrelation",
    "arch_lm_p20": "remaining ARCH effect",
    "sign_bias_p": "remaining sign bias",
}

DATA_COLOR = "#2f6f9f"
REFERENCE_COLOR = "#6b6b6b"


def describe_returns(train_returns: pd.Series, all_returns: pd.Series) -> pd.DataFrame:
    rows = []
    for sample, series in {"train": train_returns, "full": all_returns}.items():
        values = series.to_numpy(dtype=float)
        jb_stat, jb_p_value, skewness, kurtosis = jarque_bera(values)
        rows.append({
            "sample": sample,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "jarque_bera": jb_stat,
            "jarque_bera_p_value": jb_p_value,
        })
    return pd.DataFrame(rows)


def sign_bias_test(residuals: np.ndarray, innovations: np.ndarray | None = None) -> tuple[float, float]:
    """Joint sign and size bias test of Engle and Ng (NBER w3681, Eq. 18)."""
    previous = (residuals if innovations is None else innovations)[:-1]
    negative = (previous < 0).astype(float)
    regressors = sm.add_constant(np.column_stack([
        negative,
        negative * previous,
        (1.0 - negative) * previous,
    ]))
    fit = sm.OLS(residuals[1:] ** 2, regressors).fit()
    return float(fit.fvalue), float(fit.f_pvalue)


def heteroskedasticity_tests(residuals: np.ndarray, innovations: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for series_name, series in {"level": residuals, "squared": residuals ** 2}.items():
        ljung_box = acorr_ljungbox(series, lags=list(LJUNG_BOX_LAGS))
        for lag, row in ljung_box.iterrows():
            rows.append({
                "test": "Ljung-Box",
                "series": series_name,
                "lags": lag,
                "statistic": row["lb_stat"],
                "p_value": row["lb_pvalue"],
            })
    for lag in ARCH_LM_LAGS:
        statistic, p_value, _, _ = het_arch(residuals, nlags=lag)
        rows.append({"test": "ARCH-LM", "series": "level", "lags": lag, "statistic": statistic, "p_value": p_value})

    statistic, p_value = sign_bias_test(residuals, innovations)
    rows.append({"test": "Sign bias", "series": "level", "lags": 1, "statistic": statistic, "p_value": p_value})
    return pd.DataFrame(rows)


def _persistence(params: pd.Series) -> float:
    if "d" in params.index:
        return np.nan
    total = 0.0
    for name, value in params.items():
        if name.startswith(("alpha", "beta")):
            total += value
        elif name.startswith("gamma"):
            total += value / 2
    return total


def _standardized_residuals(fitted) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(fitted.std_resid, dtype=float)
    innovations = np.asarray(fitted.resid, dtype=float)
    valid = ~np.isnan(z)
    return z[valid], innovations[valid]


def _candidate_row(fitted) -> dict:
    volatility_params = [name for name in fitted.params.index if name not in DISTRIBUTION_PARAMS]
    z, innovations = _standardized_residuals(fitted)
    return {
        "converged": fitted.convergence_flag == 0,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "log_likelihood": fitted.loglikelihood,
        "persistence": _persistence(fitted.params),
        "d": fitted.params.get("d", np.nan),
        "non_significant_coefficients": int((fitted.pvalues[volatility_params] > SIGNIFICANCE).sum()),
        "ljung_box_z_p20": acorr_ljungbox(z, lags=[20])["lb_pvalue"].iloc[0],
        "ljung_box_z2_p20": acorr_ljungbox(z ** 2, lags=[20])["lb_pvalue"].iloc[0],
        "arch_lm_p20": het_arch(z, nlags=20)[1],
        "sign_bias_p": sign_bias_test(z, innovations)[1],
    }


def rejection_reasons(row: dict) -> list[str]:
    if not row.get("converged"):
        return ["did not converge"]

    reasons = []
    if row["non_significant_coefficients"] > 0:
        reasons.append("non-significant coefficients")
    if row["persistence"] >= MAX_PERSISTENCE:
        reasons.append("persistence not below 1")
    if not np.isnan(row["d"]) and not 0 < row["d"] < 1:
        reasons.append("d outside (0, 1)")
    for column, label in RESIDUAL_RULES.items():
        if row[column] <= SIGNIFICANCE:
            reasons.append(label)
    return reasons


def compare_candidates(model, residuals: np.ndarray) -> pd.DataFrame:
    justifications = {}
    if model.selection is not None:
        justifications = {
            json.dumps(candidate.hyperparameters, sort_keys=True): candidate.justification
            for candidate in model.selection.candidates
        }

    rows = []
    for hyperparameters in model.config.hyperparameters_list:
        justification = justifications.get(json.dumps(hyperparameters, sort_keys=True))
        row = {
            "vol": hyperparameters.get("vol", "GARCH"),
            "p": hyperparameters.get("p", 1),
            "o": hyperparameters.get("o", 0),
            "q": hyperparameters.get("q", 1),
            "dist": hyperparameters.get("dist", "normal"),
            "preselected": justification is not None,
            "chosen": hyperparameters == model.best_hyperparameters,
            "justification": justification or "",
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.build_arch_model(residuals * model.scale, hyperparameters).fit(
                    disp="off", options={"maxiter": 500}
                )
            row.update(_candidate_row(fitted))
        except Exception as e:
            logger.warning(f"Could not evaluate candidate {hyperparameters}: {e}")
            row["converged"] = False
        reasons = rejection_reasons(row)
        row["admissible"] = not reasons
        row["rejection_reason"] = "; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def selection_table(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = ["vol", "p", "o", "q", "dist", "aic", "bic", "admissible", "chosen", "justification"]
    return candidates.loc[candidates["preselected"] | candidates["chosen"], columns]


def coefficients_table(fitted) -> pd.DataFrame:
    return pd.DataFrame({
        "coefficient": fitted.params,
        "std_error": fitted.std_err,
        "t": fitted.tvalues,
        "p_value": fitted.pvalues,
    }).rename_axis("parameter").reset_index()


def _plot_acf(axis, values: np.ndarray, title: str, lags: int = ACF_LAGS) -> None:
    correlations = acf(values, nlags=lags, fft=True)[1:]
    positions = np.arange(1, len(correlations) + 1)
    band = 1.96 / np.sqrt(len(values))
    axis.axhspan(-band, band, color=REFERENCE_COLOR, alpha=0.15, linewidth=0)
    axis.axhline(0, color=REFERENCE_COLOR, linewidth=0.8)
    axis.vlines(positions, 0, correlations, color=DATA_COLOR, linewidth=1.0)
    axis.set_title(title, fontsize=11, loc="left")
    axis.set_xlabel("Rezago en barras de 15 minutos (banda gris: 95 %)")
    axis.set_ylabel("Autocorrelación")
    axis.set_xlim(0, len(correlations) + 1)
    axis.spines[["top", "right"]].set_visible(False)


def _use_decimal_comma(fig) -> None:
    formatter = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",").replace("-", "\N{MINUS SIGN}"))
    for axis in fig.axes:
        axis.xaxis.set_major_formatter(formatter)
        axis.yaxis.set_major_formatter(formatter)


def plot_returns_acf(returns: np.ndarray, symbol: str, path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5))
    _plot_acf(axes[0], returns, f"{symbol}: FAC de los retornos")
    _plot_acf(axes[1], returns ** 2, f"{symbol}: FAC de los retornos al cuadrado")
    _use_decimal_comma(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_standardized_residuals(z: np.ndarray, symbol: str, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    histogram = axes[0, 0]
    limit = np.quantile(np.abs(z), 0.999)
    grid = np.linspace(-limit, limit, 300)
    histogram.hist(z, bins=np.linspace(-limit, limit, 60), density=True, color=DATA_COLOR, alpha=0.75, label="Residuos")
    histogram.plot(grid, stats.norm.pdf(grid), color=REFERENCE_COLOR, linewidth=1.5, label="Normal estándar")
    histogram.set_title(f"{symbol}: distribución de los residuos estandarizados", fontsize=11, loc="left")
    histogram.set_xlabel("Residuo estandarizado")
    histogram.set_ylabel("Densidad")
    histogram.spines[["top", "right"]].set_visible(False)
    histogram.legend(frameon=False, fontsize=9)

    qq = axes[0, 1]
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(z, dist="norm")
    qq.scatter(theoretical, ordered, s=8, color=DATA_COLOR, label="Residuos")
    qq.plot(theoretical, slope * theoretical + intercept, color=REFERENCE_COLOR, linewidth=1.2, label="Referencia normal")
    qq.set_title("Gráfico QQ frente a la normal", fontsize=11, loc="left")
    qq.set_xlabel("Cuantil teórico")
    qq.set_ylabel("Cuantil observado")
    qq.spines[["top", "right"]].set_visible(False)
    qq.legend(frameon=False, fontsize=9)

    _plot_acf(axes[1, 0], z, "FAC de los residuos estandarizados")
    _plot_acf(axes[1, 1], z ** 2, "FAC de los residuos estandarizados al cuadrado")

    _use_decimal_comma(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_garch_diagnostics(model, output_path: str) -> str:
    if not model.is_fitted:
        raise ValueError("Model must be fitted before computing diagnostics.")

    path = os.path.join(output_path, "garch_diagnostics", model.symbol)
    os.makedirs(path, exist_ok=True)

    train_returns = model.train_returns.to_numpy(dtype=float)
    residuals = np.asarray(model.mean_model.resid(), dtype=float)
    z, innovations = _standardized_residuals(model.model)

    describe_returns(model.train_returns, model.returns).to_csv(os.path.join(path, "returns_summary.csv"), index=False)
    heteroskedasticity_tests(residuals).to_csv(os.path.join(path, "pre_estimation_tests.csv"), index=False)
    heteroskedasticity_tests(z, innovations).to_csv(os.path.join(path, "standardized_residual_tests.csv"), index=False)
    candidates = compare_candidates(model, residuals)
    candidates.to_csv(os.path.join(path, "candidates.csv"), index=False)
    coefficients_table(model.model).to_csv(os.path.join(path, "coefficients.csv"), index=False)
    selection_table(candidates).to_csv(os.path.join(path, "selection.csv"), index=False)
    pd.DataFrame([{
        "arima_order": str(model.mean_model.order),
        "aic": model.mean_model.aic(),
    }]).to_csv(os.path.join(path, "mean_model.csv"), index=False)

    plot_returns_acf(train_returns, model.symbol, os.path.join(path, "returns_acf.png"))
    plot_standardized_residuals(z, model.symbol, os.path.join(path, "standardized_residuals.png"))

    logger.info(f"GARCH diagnostics saved: {path}")
    return path
