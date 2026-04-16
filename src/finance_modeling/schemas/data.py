from __future__ import annotations

import os
import pandas as pd
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from finance_modeling.utils import get_main_root


class AssetType(StrEnum):
    FOREX = "forex"
    INDEX = "index"
    CRYPTO = "crypto"


class AssetMetadata(BaseModel):
    symbol: str
    asset_type: AssetType
    description: str = ""
    column_to_use: str = "close"
    data_folder: str = Field(exclude=True)
    active: bool = True
    data_path: str = Field(
        default_factory=lambda: os.path.join("data", "files"),
        exclude=True,
    )


class ListOfAssets(BaseModel):
    assets: list[AssetMetadata]


class TimeSeriesInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    metadata: AssetMetadata
    series: pd.Series
    log_returns: pd.Series
    frequency: str = "15min"
    train: pd.Series = Field(default_factory=lambda: pd.Series(dtype=float))
    test: pd.Series = Field(default_factory=lambda: pd.Series(dtype=float))
    split_index: int = 0
