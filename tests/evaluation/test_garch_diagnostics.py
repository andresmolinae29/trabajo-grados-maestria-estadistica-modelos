from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance_modeling.evaluation.garch_diagnostics import (
    compare_candidates,
    heteroskedasticity_tests,
    rejection_reasons,
    save_garch_diagnostics,
    sign_bias_test,
)
from finance_modeling.models.garch import GARCHModel
from finance_modeling.schemas import GarchSelection, ModelConfig


def simulate_garch_returns(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    variance = 1e-5
    for t in range(1, n):
        variance = 1e-6 + 0.15 * returns[t - 1] ** 2 + 0.75 * variance
        returns[t] = np.sqrt(variance) * rng.standard_normal()
    return returns


CANDIDATES = [
    {"p": 1, "q": 1, "dist": "normal"},
    {"p": 1, "o": 1, "q": 1, "dist": "normal"},
    {"vol": "FIGARCH", "p": 1, "q": 1, "dist": "normal"},
]


def make_fitted_model(selection: GarchSelection | None = None) -> GARCHModel:
    n = 900
    index = pd.date_range("2026-01-01", periods=n, freq="15min")
    prices = pd.Series(100 * np.exp(np.cumsum(simulate_garch_returns(n))), index=index)
    model = GARCHModel(
        config=ModelConfig(name="GARCH", hyperparameters_list=CANDIDATES),
        symbol="SIM",
        prices=prices,
        selection=selection,
    )
    return model.fit()  # type: ignore


def test_arch_tests_detect_volatility_clustering_only_in_clustered_series() -> None:
    clustered = heteroskedasticity_tests(simulate_garch_returns(3000))
    white_noise = heteroskedasticity_tests(np.random.default_rng(3).standard_normal(3000))

    def arch_lm_p(table: pd.DataFrame) -> float:
        return table.query("test == 'ARCH-LM' and lags == 10")["p_value"].iloc[0]

    assert arch_lm_p(clustered) < 0.01
    assert arch_lm_p(white_noise) > 0.01


def test_sign_bias_detects_asymmetric_response() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    returns = np.zeros(n)
    for t in range(1, n):
        scale = 2.0 if returns[t - 1] < 0 else 0.5
        returns[t] = scale * rng.standard_normal()

    assert sign_bias_test(returns)[1] < 0.01
    assert sign_bias_test(rng.standard_normal(n))[1] > 0.01


def test_sign_bias_uses_lagged_innovations_when_given() -> None:
    rng = np.random.default_rng(2)
    innovations = rng.standard_normal(2000)
    z = np.ones(2000)
    z[1:][innovations[:-1] < 0] = 3.0

    assert sign_bias_test(z, innovations)[1] < 0.01
    assert sign_bias_test(z, rng.standard_normal(2000))[1] > 0.01


def _candidate(**changes) -> dict:
    row = {
        "converged": True,
        "non_significant_coefficients": 0,
        "persistence": 0.95,
        "d": np.nan,
        "ljung_box_z_p20": 0.5,
        "ljung_box_z2_p20": 0.5,
        "arch_lm_p20": 0.5,
        "sign_bias_p": 0.5,
    }
    return {**row, **changes}


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, []),
        ({"persistence": np.nan, "d": 0.4}, []),
        ({"converged": False}, ["did not converge"]),
        ({"non_significant_coefficients": 2}, ["non-significant coefficients"]),
        ({"persistence": 1.0}, ["persistence not below 1"]),
        ({"persistence": np.nan, "d": 1.2}, ["d outside (0, 1)"]),
        ({"arch_lm_p20": 0.01}, ["remaining ARCH effect"]),
        (
            {"ljung_box_z_p20": 0.01, "sign_bias_p": 0.01},
            ["residual autocorrelation", "remaining sign bias"],
        ),
    ],
)
def test_rejection_rules(changes: dict, expected: list[str]) -> None:
    assert rejection_reasons(_candidate(**changes)) == expected


def test_save_records_preselected_models_and_their_justifications(tmp_path: Path) -> None:
    selection = GarchSelection(
        symbol="SIM",
        candidates=[
            {"hyperparameters": CANDIDATES[0], "justification": "Razón A."},
            {"hyperparameters": CANDIDATES[1], "justification": "Razón B."},
        ],
    )

    path = Path(save_garch_diagnostics(make_fitted_model(selection), str(tmp_path)))

    candidates = pd.read_csv(path / "candidates.csv")
    assert list(candidates["preselected"]) == [True, True, False]
    assert not candidates["chosen"].iloc[2]
    table = pd.read_csv(path / "selection.csv")
    assert list(table["justification"]) == ["Razón A.", "Razón B."]
    assert table["chosen"].sum() == 1
    assert table.loc[table["chosen"], "aic"].iloc[0] == pytest.approx(table["aic"].min())


def test_save_requires_fitted_model(tmp_path: Path) -> None:
    index = pd.date_range("2026-01-01", periods=400, freq="15min")
    prices = pd.Series(100 * np.exp(np.cumsum(simulate_garch_returns(400))), index=index)
    model = GARCHModel(ModelConfig(name="GARCH"), "SIM", prices=prices)

    with pytest.raises(ValueError, match="fitted"):
        save_garch_diagnostics(model, str(tmp_path))


def test_save_writes_tables_and_figures_for_all_candidates(tmp_path: Path) -> None:
    model = make_fitted_model()

    path = Path(save_garch_diagnostics(model, str(tmp_path)))

    assert sorted(p.name for p in path.iterdir()) == [
        "candidates.csv",
        "coefficients.csv",
        "mean_model.csv",
        "pre_estimation_tests.csv",
        "returns_acf.png",
        "returns_summary.csv",
        "selection.csv",
        "standardized_residual_tests.csv",
        "standardized_residuals.png",
    ]

    candidates = pd.read_csv(path / "candidates.csv")
    assert list(candidates["vol"]) == ["GARCH", "GARCH", "FIGARCH"]
    assert candidates["chosen"].sum() == 1
    assert candidates.loc[candidates["chosen"], "aic"].iloc[0] == pytest.approx(candidates["aic"].min())
    assert candidates["admissible"].dtype == bool
    assert not candidates["preselected"].any()
    assert len(pd.read_csv(path / "selection.csv")) == 1
    assert candidates.loc[candidates["vol"] == "FIGARCH", "d"].notna().all()
    assert candidates.loc[candidates["vol"] == "GARCH", "persistence"].notna().all()

    summary = pd.read_csv(path / "returns_summary.csv")
    assert list(summary["sample"]) == ["train", "full"]
    assert (summary["kurtosis"] > 0).all()


def test_compare_candidates_marks_candidates_that_fail_to_fit() -> None:
    def failing_build(residuals, hyperparameters):
        raise RuntimeError("cannot fit")

    model = SimpleNamespace(
        selection=None,
        config=ModelConfig(name="GARCH", hyperparameters_list=CANDIDATES[:1]),
        best_hyperparameters=CANDIDATES[0],
        scale=1.0,
        build_arch_model=failing_build,
    )

    table = compare_candidates(model, np.zeros(10))

    assert table["converged"].tolist() == [False]
    assert table["admissible"].tolist() == [False]
    assert table["rejection_reason"].tolist() == ["did not converge"]
