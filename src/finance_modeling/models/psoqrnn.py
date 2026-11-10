from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.nn.utils import vector_to_parameters

from .base import BaseVolatilityModel

from ..schemas import ModelConfig
from ..utils import logger


np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PSO_HYPERPARAMETERS = ("n_particles", "max_iter", "inertia", "c1", "c2", "v_max", "init")


def quantile_loss(y_true: torch.Tensor,
                  y_pred: torch.Tensor,
                  tau: float) -> torch.Tensor:
    u = y_true - y_pred
    return torch.where(u >= 0, tau * u, (tau - 1) * u).mean()


class QRNN(nn.Module):

    def __init__(self, n_hidden: int = 3):
        super().__init__()
        self.hidden = nn.Linear(2, n_hidden)
        self.output = nn.Linear(n_hidden, 1)

    def forward(self, x) -> torch.Tensor:
        h = torch.tanh(self.hidden(x))
        return torch.tanh(self.output(h))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_weights(self) -> np.ndarray:
        return np.concatenate([p.detach().numpy().ravel()
                                for p in self.parameters()])

    def set_weights(self, w: np.ndarray) -> None:
        vector_to_parameters(torch.from_numpy(w).float(), self.parameters())


def qrnn_swarm_forward(weights: torch.Tensor, X: torch.Tensor, n_hidden: int) -> torch.Tensor:
    # One QRNN per row of weights, with the parameter order of QRNN.set_weights.
    n = weights.shape[0]
    w1 = weights[:, :2 * n_hidden].reshape(n, n_hidden, 2)
    b1 = weights[:, 2 * n_hidden:3 * n_hidden]
    w2 = weights[:, 3 * n_hidden:4 * n_hidden]
    b2 = weights[:, 4 * n_hidden]
    hidden = torch.tanh(torch.einsum("nk,phk->pnh", X, w1) + b1[:, None, :])
    return torch.tanh(torch.einsum("pnh,ph->pn", hidden, w2) + b2[:, None])


class PSO:
    """
    PSO estándar. Hiperparámetros tomados directamente de la Sección 5.3:
        γ  (inercia)   = 0.8
        C1 = C2        = 2.0
        Vmin / Vmax    = ±5
        Tamaño enjambre= 60
        Iteraciones    = 5000
        τ (cuantil)    = 0.5   ← mejor resultado en el paper
    """
    def __init__(
            self,
            n_particles: int = 60,
            inertia: float = 0.8,
            c1: float = 2.0,
            c2: float = 2.0,
            v_min: float = -5.0,
            v_max: float = 5.0,
            max_iter: int = 5000,
            tau: float = 0.5,
            verbose_every: int = 500,
            init: str = "unit"
            ):

        self.n_particles = n_particles
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.v_min = v_min
        self.v_max = v_max
        self.max_iter = max_iter
        self.tau = tau
        self.verbose_every = verbose_every
        self.init = init

    def _fitness(self, w: np.ndarray, model: QRNN, X: torch.Tensor, y: torch.Tensor) -> float:
        model.set_weights(w)
        with torch.no_grad():
            return quantile_loss(y, model(X), self.tau).item()

    def _swarm_fitness(self, swarm: np.ndarray, X: torch.Tensor, y: torch.Tensor, n_hidden: int) -> np.ndarray:
        weights = torch.as_tensor(swarm, dtype=torch.float32, device=X.device)
        with torch.no_grad():
            u = y - qrnn_swarm_forward(weights, X, n_hidden)
            return torch.where(u >= 0, self.tau * u, (self.tau - 1) * u).mean(dim=1).cpu().numpy().astype(np.float64)

    def optimize(self, model: QRNN, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:

        D = model.n_params()
        n_hidden = model.hidden.out_features
        X = torch.as_tensor(X_train, device=DEVICE)
        y = torch.as_tensor(y_train, device=DEVICE)

        # Inicialización aleatoria de partículas
        low = 0.0 if self.init == "unit" else -1.0
        swarm = np.random.uniform(low, 1.0, (self.n_particles, D))
        velocities = np.random.uniform(self.v_min, self.v_max, (self.n_particles, D))

        personal_best = swarm.copy()
        personal_best_fit = self._swarm_fitness(swarm, X, y, n_hidden)
        best = int(np.argmin(personal_best_fit))
        global_best = personal_best[best].copy()
        global_best_fit = personal_best_fit[best]

        for it in range(1, self.max_iter + 1):

            r1 = np.random.rand(self.n_particles, D)
            r2 = np.random.rand(self.n_particles, D)

            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best - swarm) +
                            self.c2 * r2 * (global_best - swarm))

            velocities = np.clip(velocities, self.v_min, self.v_max)
            swarm += velocities

            fit = self._swarm_fitness(swarm, X, y, n_hidden)
            improved = fit < personal_best_fit
            personal_best[improved] = swarm[improved]
            personal_best_fit[improved] = fit[improved]

            best = int(np.argmin(personal_best_fit))
            if personal_best_fit[best] < global_best_fit:
                global_best_fit = personal_best_fit[best]
                global_best = personal_best[best].copy()

            if it % self.verbose_every == 0 or it == self.max_iter:
                logger.info(f"PSO Iteration {it}/{self.max_iter} - Best Loss: {global_best_fit:.6f}")

        return global_best


class PSOQRNNModel(BaseVolatilityModel):

    name = "PSO-QRNN"

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):
        super().__init__(config, symbol, **kwargs)
        self.qrnn = QRNN(n_hidden=kwargs.get("n_hidden", 3))
        pso_kwargs = {key: kwargs[key] for key in PSO_HYPERPARAMETERS if key in kwargs}
        if "v_max" in pso_kwargs:
            pso_kwargs["v_min"] = -pso_kwargs["v_max"]
        self.pso = PSO(**pso_kwargs)

        self.innovation = kwargs.get("innovation", pd.Series())
        self.volatility = kwargs.get("volatility", pd.Series())
        self.returns = kwargs.get("returns", pd.Series())

        X, y = self._build_causal_dataset() if config.causal else self._build_dataset()

        self.X_train, self.y_train, self.X_test, self.y_test = self._train_test_split(X, y)

    def _build_dataset(
            self,
            omega: int = 252
        ) -> tuple[np.ndarray, np.ndarray]:

        M = len(self.volatility)
        sig_prev = self.volatility.iloc[:-1]
        eps_curr = self.innovation[omega: omega + M - 1]
        sig_curr = self.volatility[1:]

        X = np.column_stack([sig_prev.to_numpy(), eps_curr.to_numpy()]).astype(np.float32)
        y = sig_curr.to_numpy().astype(np.float32)

        return X, y # type: ignore

    def _build_causal_dataset(
            self,
            omega: int = 252,
            train_size: float = 0.8
        ) -> tuple[np.ndarray, np.ndarray]:

        # ε del último retorno de la ventana de σ_{t-1}, centrado con la media de entrenamiento.
        M = len(self.volatility)
        r = self.returns.to_numpy(dtype=float)
        if len(r) < M + omega - 2:
            raise ValueError("PSOQRNN causal requiere los retornos de toda la ventana de la volatilidad.")
        split = int((M - 1) * train_size)
        eps_prev = r[omega - 1:omega - 1 + M - 1] - r[:split + omega].mean()

        X = np.column_stack([self.volatility.iloc[:-1].to_numpy(), eps_prev]).astype(np.float32)
        y = self.volatility.iloc[1:].to_numpy().astype(np.float32)

        return X, y

    def _train_test_split(self, X: np.ndarray, y: np.ndarray, train_size: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(X) * train_size)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    def fit(self) -> PSOQRNNModel:

        nh = self.qrnn.hidden.out_features
        logger.info(f"[PSOQRNN] n_hidden={nh}, n_params={self.qrnn.n_params()}")
        best_weights = self.pso.optimize(self.qrnn, self.X_train, self.y_train)
        self.qrnn.set_weights(best_weights)
        self.is_fitted = True
        return self

    def predict(self) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        self.qrnn.eval()
        with torch.no_grad():
            return self.qrnn(torch.as_tensor(self.X_test)).numpy().ravel()
