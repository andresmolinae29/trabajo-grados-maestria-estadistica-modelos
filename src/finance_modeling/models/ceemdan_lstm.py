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
from ..utils import logger


np.random.seed(42)
torch.manual_seed(42)


def ceemdan_features(series):

    ceemdan = CEEMDAN()
    imfs = ceemdan(series.values)

    return imfs.T


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

    def __init__(self, n_features: int):

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
        X, y = self._build_raw_dataset()
        self.train_loader, self.test_loader, self.x_scaler, self.y_scaler = self._build_dataset(X, y)
        self.y_test = self.y_scaler.inverse_transform(self.test_loader.dataset.y.cpu().numpy().reshape(-1,1)) # type: ignore
        self.model = CEEMDANLSTM(n_features=X.shape[1]).to(self.DEVICE)

    def _build_raw_dataset(self) -> tuple[np.ndarray, np.ndarray]:

        imfs = ceemdan_features(self.returns)
        T = min(len(self.volatility), imfs.shape[0])

        sig = self.volatility.values[-T:]
        imfs = imfs[-T:]

        X_raw = []
        y_raw = []

        for i in range(T - 1):

            x_i = np.concatenate([
                imfs[i],
                [sig[i]]
            ])

            y_i = sig[i + 1]

            X_raw.append(x_i)
            y_raw.append(y_i)

        return np.array(X_raw), np.array(y_raw)

    def _build_dataset(self, X_raw, y_raw) -> tuple[DataLoader, DataLoader, MinMaxScaler, MinMaxScaler]:

        split = int(len(X_raw) * 0.8)
        X_train, X_test = X_raw[:split], X_raw[split:]
        y_train, y_test = y_raw[:split], y_raw[split:]

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)

        y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
        y_test = y_scaler.transform(y_test.reshape(-1,1))

        X_train, y_train = create_sequences(X_train, y_train, self.WINDOW_SIZE)
        X_test, y_test = create_sequences(X_test, y_test, self.WINDOW_SIZE)

        train_loader = DataLoader(VolDataset(X_train, y_train), batch_size=self.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(VolDataset(X_test, y_test), batch_size=self.BATCH_SIZE, shuffle=False)

        return train_loader, test_loader, x_scaler, y_scaler

    def fit(self) -> BaseVolatilityModel:

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)

        n_epochs = self.EPOCHS
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