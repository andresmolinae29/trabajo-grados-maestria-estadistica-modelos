import arch
import numpy as np
import pmdarima as pm
import pandas as pd
from copy import deepcopy

from .base import BaseVolatilityModel
from ..schemas import ModelConfig
from ..utils import logger

import warnings
from arch.utility.exceptions import ConvergenceWarning


class GARCHModel(BaseVolatilityModel):

    name = "GARCH"
    SCALE = 100.0  # escala fija para mejorar condicionamiento numérico del optimizador

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):
        super().__init__(config, symbol, **kwargs)

        prices: pd.Series = kwargs.get("prices", pd.Series())

        # Log-retornos internos: ln(P_t / P_{t-1}) — necesario para GARCH.
        # Los retornos del preprocesador son para otros modelos; GARCH los calcula
        # desde los precios para garantizar consistencia interna.
        if prices.empty:
            raise ValueError("GARCHModel requiere 'prices' para calcular log-retornos.")
        self.returns: pd.Series = pd.Series(
            np.log(prices / prices.shift(1)), dtype=float
        ).dropna()

        # Volatilidad realizada de referencia (y_test) calculada desde los mismos
        # log-retornos que usa GARCH, para que la comparación sea consistente.
        OMEGA = 252
        realized_vol: pd.Series = (
            self.returns.rolling(window=OMEGA).std(ddof=1) * np.sqrt(252)
        ).dropna()

        split = int(len(realized_vol) * 0.8)
        vol_train = realized_vol.iloc[:split]
        vol_test = realized_vol.iloc[split:]
        self.y_test: np.ndarray = vol_test.to_numpy()

        self.train_returns = self.returns.loc[:vol_train.index[-1]]
        self.test_returns = self.returns.loc[vol_test.index]

        self.best_hyperparameters: dict = {}
        self.mean_model = None

    def __fit_mean_model(self, X: pd.Series):
        return pm.auto_arima(
            X.to_numpy(dtype=float),
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

    def __select_hyperparameters(self, residuals: pd.Series) -> dict:
        best_aic = float("inf")
        best_hyperparameters: dict = {}

        for hyperparameters in self.config.hyperparameters_list:
            try:
                logger.info(f"Evaluating GARCH hyperparameters: {hyperparameters}")
                model = arch.arch_model(
                    residuals * self.SCALE,
                    vol="GARCH",
                    rescale=False,
                    mean="Zero",
                    **hyperparameters,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    fitted_model = model.fit(
                        disp="off",
                        options={"maxiter": 500},
                    )
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    logger.warning(
                        f"GARCH no convergió con {hyperparameters}, descartando."
                    )
                    continue

                aic = fitted_model.aic
                if aic < best_aic:
                    best_aic = aic
                    best_hyperparameters = hyperparameters
            except Exception as e:
                logger.warning(
                    f"Error fitting GARCH model with hyperparameters {hyperparameters}: {e}"
                )
                continue

        if not best_hyperparameters:
            raise RuntimeError(
                "No se pudo ajustar ningún modelo GARCH con los hyperparámetros provistos."
            )

        return best_hyperparameters

    def fit(self) -> BaseVolatilityModel:
        self.mean_model = self.__fit_mean_model(self.train_returns)
        train_residuals = pd.Series(self.mean_model.resid(), dtype=float)

        self.best_hyperparameters = self.__select_hyperparameters(train_residuals)
        logger.info(f"Selected GARCH hyperparameters: {self.best_hyperparameters}")

        model = arch.arch_model(
            train_residuals * self.SCALE,
            vol="GARCH",
            rescale=False,
            mean="Zero",
            **self.best_hyperparameters,
        )
        self.model = model.fit(disp="off")
        self.is_fitted = True
        return self

    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        mean_model = deepcopy(self.mean_model)
        test_arima_resids: list[float] = []
        for index, (_, actual_value) in enumerate(self.test_returns.items()):
            if index % 100 == 0:
                logger.info(f"Computing ARIMA residual {index}/{len(self.test_returns)}")
            mean_fc = float(np.asarray(mean_model.predict(n_periods=1), dtype=float)[0])  # type: ignore
            test_arima_resids.append(float(actual_value) - mean_fc)
            mean_model.update([float(actual_value)])  # type: ignore

        # am.fix fija los parámetros estimados en entrenamiento y computa la
        # volatilidad condicional de forma recursiva sobre toda la serie.
        # GARCH(p,q): σ_t² = ω + Σαᵢ ε²_{t-i} + Σβⱼ σ²_{t-j}
        # → conditional_volatility[t] solo usa información hasta t-1 (1-step-ahead).
        train_resids = np.asarray(self.model.resid, dtype=float)  # ya en escala SCALE
        test_resids_scaled = np.array(test_arima_resids) * self.SCALE
        all_resids = np.concatenate([train_resids, test_resids_scaled])

        am = arch.arch_model(all_resids, vol="GARCH", rescale=False, mean="Zero", **self.best_hyperparameters)
        fixed_res = am.fix(self.model.params)

        n_train = len(train_resids)
        cond_vol_test = np.asarray(fixed_res.conditional_volatility, dtype=float)[n_train:]

        return (cond_vol_test / self.SCALE) * np.sqrt(252)
