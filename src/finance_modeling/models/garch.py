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
    OMEGA = 252  # ventana de la variable objetivo

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):
        super().__init__(config, symbol, **kwargs)

        prices: pd.Series = kwargs.get("prices", pd.Series())

        # Misma definición de retornos de los demás modelos: (P_t - P_{t-1}) / P_t.
        if prices.empty:
            raise ValueError("GARCHModel requiere 'prices' para calcular los retornos.")
        self.returns: pd.Series = pd.Series(
            (prices - prices.shift(1)) / prices, dtype=float
        ).dropna()

        # La variable objetivo es la misma de los demás modelos cuando se recibe.
        volatility: pd.Series = kwargs.get("volatility", pd.Series())
        realized_vol: pd.Series = volatility if not volatility.empty else (
            self.returns.rolling(window=self.OMEGA).std(ddof=1) * np.sqrt(252)
        ).dropna()

        split = int(len(realized_vol) * 0.8)
        vol_train = realized_vol.iloc[:split]
        vol_test = realized_vol.iloc[split:]
        self.y_test: np.ndarray = vol_test.to_numpy()

        self.train_returns = self.returns.loc[:vol_train.index[-1]]
        self.test_returns = self.returns.loc[vol_test.index]

        self.selection = kwargs.get("selection")
        self.scale: float = 1.0
        self.best_hyperparameters: dict = {}
        self.mean_model = None

    @staticmethod
    def build_arch_model(residuals, hyperparameters: dict):
        return arch.arch_model(residuals, rescale=False, mean="Zero", **{"vol": "GARCH", **hyperparameters})

    def __fit_mean_model(self, X: pd.Series):
        return pm.auto_arima(
            X.to_numpy(dtype=float),
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

    def __candidates(self) -> list[dict]:
        if self.selection is None:
            return self.config.hyperparameters_list

        candidates = [candidate.hyperparameters for candidate in self.selection.candidates]
        for hyperparameters in candidates:
            if hyperparameters not in self.config.hyperparameters_list:
                raise ValueError(f"Preselected model for {self.symbol} is not a candidate: {hyperparameters}")
        return candidates

    def __select_hyperparameters(self, residuals: pd.Series) -> dict:
        best_aic = float("inf")
        best_hyperparameters: dict = {}

        for hyperparameters in self.__candidates():
            try:
                logger.info(f"Evaluating GARCH hyperparameters: {hyperparameters}")
                model = self.build_arch_model(residuals * self.scale, hyperparameters)
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

        # Con una escala fija, el optimizador se detiene cerca de sus valores iniciales.
        self.scale = 1.0 / float(train_residuals.std())

        self.best_hyperparameters = self.__select_hyperparameters(train_residuals)
        logger.info(f"Selected GARCH hyperparameters: {self.best_hyperparameters}")

        model = self.build_arch_model(train_residuals * self.scale, self.best_hyperparameters)
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
        train_resids = np.asarray(self.model.resid, dtype=float)  # ya en escala self.scale
        test_resids_scaled = np.array(test_arima_resids) * self.scale
        all_resids = np.concatenate([train_resids, test_resids_scaled])

        am = self.build_arch_model(all_resids, self.best_hyperparameters)
        fixed_res = am.fix(self.model.params)

        n_train = len(train_resids)
        cond_vol = np.asarray(fixed_res.conditional_volatility, dtype=float) / self.scale
        self.point_predictions: np.ndarray = cond_vol[n_train:] * np.sqrt(252)

        # Promedia las varianzas condicionales de la misma ventana de la variable objetivo.
        window_var = pd.Series(cond_vol ** 2).rolling(self.OMEGA).mean().to_numpy()[n_train:]

        return np.sqrt(window_var) * np.sqrt(252)
