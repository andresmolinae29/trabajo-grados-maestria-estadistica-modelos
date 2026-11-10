import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

from PyEMD import CEEMDAN
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .base import BaseVolatilityModel

from ..schemas import (
    ModelConfig,
)
from ..utils import get_main_root, logger


np.random.seed(42)
torch.manual_seed(42)

CAUSAL_WINDOW = 512
CAUSAL_MAX_IMF = 6
CAUSAL_SEED = 42


def ceemdan_features(series, seed: int = 42):

    ceemdan = CEEMDAN(seed=seed)
    imfs = ceemdan(series.values)

    return imfs.T


def last_components(window: np.ndarray, max_imf: int, seed: int) -> np.ndarray:
    components = CEEMDAN(seed=seed, parallel=False)(window, max_imf=max_imf)
    if components.shape[0] < max_imf + 1:
        missing = np.zeros((max_imf + 1 - components.shape[0], components.shape[1]))
        components = np.vstack([components[:-1], missing, components[-1:]])
    return components[:, -1]


def causal_ceemdan_features(
        series: pd.Series,
        symbol: str,
        window: int = CAUSAL_WINDOW,
        max_imf: int = CAUSAL_MAX_IMF,
        seed: int = CAUSAL_SEED,
        processes: int | None = None,
        cache_dir: str | None = None,
    ) -> np.ndarray:

    # Fila t: último punto de cada componente de la descomposición de los `window` retornos que terminan en t.
    values = series.to_numpy(dtype=float)
    cache_dir = cache_dir or os.path.join(get_main_root(), "results", "causal_ceemdan")
    cache_file = os.path.join(cache_dir, f"{symbol}.npz")
    settings = np.array([window, max_imf, seed])

    if os.path.exists(cache_file):
        cache = np.load(cache_file)
        cached = cache["returns"]
        if (np.array_equal(cache["settings"], settings) and len(cached) >= len(values)
                and np.allclose(cached[:len(values)], values)):
            return cache["features"][:len(values)]

    features = np.full((len(values), max_imf + 1), np.nan)
    windows = (values[end - window:end] for end in range(window, len(values) + 1))
    worker = partial(last_components, max_imf=max_imf, seed=seed)
    processes = processes or max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"[{symbol}] causal CEEMDAN over {len(values) - window + 1} windows with {processes} processes")

    if processes == 1:
        rows = map(worker, windows)
        for i, row in enumerate(rows):
            features[window - 1 + i] = row
    else:
        with Pool(processes=processes) as pool:
            for i, row in enumerate(pool.imap(worker, windows, chunksize=32)):
                features[window - 1 + i] = row
                if (i + 1) % 1000 == 0:
                    logger.info(f"[{symbol}] causal CEEMDAN {i + 1}/{len(values) - window + 1}")

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file, features=features, returns=values, settings=settings)
    return features


def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:

    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


class VolDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X).float()
        self.y = torch.as_tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CEEMDANLSTM(nn.Module):

    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2, dense: int = 64):

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, dense),
            nn.ReLU(),
            nn.Linear(dense, 1)
        )

    def forward(self, x):

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class CEEMDANLSTMModel(BaseVolatilityModel):

    name = "CEEMDAN-LSTM"
    WINDOW_SIZE = 20
    BATCH_SIZE = 32
    EPOCHS = 80
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config: ModelConfig, symbol: str, **kwargs):
        super().__init__(config, symbol, **kwargs)

        self.returns = kwargs.get("returns", pd.Series())
        self.volatility = kwargs.get("volatility", pd.Series())
        self.window_size = kwargs.get("window_size", self.WINDOW_SIZE)
        self.epochs = kwargs.get("epochs", self.EPOCHS)
        self.seed = kwargs.get("seed", 42)
        self.batch_size = kwargs.get("batch_size", self.BATCH_SIZE)
        self.lr = kwargs.get("lr", self.LR)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.shuffle = kwargs.get("shuffle", False)
        X, y = self._build_raw_dataset(kwargs.get("imfs"))
        self.train_loader, self.test_loader, self.x_scaler, self.y_scaler = self._build_dataset(X, y)
        self.y_test = self.y_scaler.inverse_transform(self.test_loader.dataset.y.cpu().numpy().reshape(-1,1)) # type: ignore
        self.model = CEEMDANLSTM(
            n_features=X.shape[1],
            hidden=kwargs.get("hidden", 128),
            layers=kwargs.get("layers", 2),
            dropout=kwargs.get("dropout", 0.2),
            dense=kwargs.get("dense", 64),
        ).to(self.DEVICE)

    def _build_raw_dataset(self, imfs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:

        if imfs is None and self.config.causal:
            imfs = causal_ceemdan_features(self.returns, self.symbol)
        elif imfs is None:
            imfs = ceemdan_features(self.returns, self.seed)
        T = min(len(self.volatility), imfs.shape[0])

        sig = self.volatility.values[-T:]
        imfs = imfs[-T:]

        X_raw = []
        y_raw = []

        for i in range(T - 1):

            # imfs[i] y sig[i + 1] son la misma barra: los retornos tienen una barra más al final.
            x_i = np.concatenate([
                imfs[i],
                [sig[i + 1]]
            ])

            y_i = sig[i + 1]

            X_raw.append(x_i)
            y_raw.append(y_i)

        return np.array(X_raw), np.array(y_raw)

    def _build_dataset(self, X_raw, y_raw) -> tuple[DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:

        split = int(len(X_raw) * 0.8)
        X_train, X_test = X_raw[:split], X_raw[split:]
        y_train, y_test = y_raw[:split], y_raw[split:]

        # La descomposición causal no existe en las primeras barras, antes de completar su ventana.
        complete = ~np.isnan(X_train).any(axis=1)
        X_train, y_train = X_train[complete], y_train[complete]

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)

        y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
        y_test = y_scaler.transform(y_test.reshape(-1,1))

        if self.config.train_context:
            # Las primeras secuencias de prueba toman como contexto las últimas filas de entrenamiento,
            # así hay pronóstico desde la primera barra de prueba.
            X_test = np.vstack([X_train[-self.window_size:], X_test])
            y_test = np.vstack([y_train[-self.window_size:], y_test])

        X_train, y_train = create_sequences(X_train, y_train, self.window_size)
        X_test, y_test = create_sequences(X_test, y_test, self.window_size)

        train_loader = DataLoader(VolDataset(X_train, y_train), batch_size=self.batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(VolDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader, x_scaler, y_scaler

    def fit(self) -> BaseVolatilityModel:

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_epochs = self.epochs
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.DEVICE), y_batch.to(self.DEVICE)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(self.train_loader):.6f}")

        self.is_fitted = True
        return self

    def predict(self) -> np.ndarray:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch, _ in self.test_loader:
                X_batch = X_batch.to(self.DEVICE)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())

        predictions = np.vstack(predictions)
        predictions = self.y_scaler.inverse_transform(predictions)

        return predictions