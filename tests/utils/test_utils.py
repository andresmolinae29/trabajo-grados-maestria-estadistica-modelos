from __future__ import annotations

import logging
from pathlib import Path

import pytest

from finance_modeling.utils.common import (
    create_experiment_directory,
    get_main_root,
    validate_file_exists,
)
from finance_modeling.utils.exceptions import DataLoaderException, ModelNotFitException
from finance_modeling.utils.logger import logger


def test_get_main_root_points_to_package_root() -> None:
    root = Path(get_main_root())

    assert root.name == "finance_modeling"
    assert (root / "config").exists()


def test_validate_file_exists_raises_for_missing_path(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="File not found at"):
        validate_file_exists(str(missing_file))


def test_create_experiment_directory_creates_nested_path(tmp_path: Path) -> None:
    created_path = create_experiment_directory(str(tmp_path), "run_001")

    assert Path(created_path).exists()
    assert Path(created_path).is_dir()
    assert Path(created_path).name == "run_001"


def test_custom_exceptions_are_standard_exceptions() -> None:
    assert issubclass(ModelNotFitException, Exception)
    assert issubclass(DataLoaderException, Exception)


def test_logger_has_console_and_file_handlers() -> None:
    handler_types = {type(handler) for handler in logger.handlers}

    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types
    assert logger.level == logging.INFO
