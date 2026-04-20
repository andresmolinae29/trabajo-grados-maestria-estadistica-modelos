from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from finance_modeling.config.config import ConfigLoader


def test_load_config_reads_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "sample.yml"
    config_path.write_text("name: demo\nvalue: 3\n", encoding="utf-8")

    ConfigLoader.load_config.cache_clear()

    loaded = ConfigLoader.load_config(str(config_path))

    assert loaded == {"name": "demo", "value": 3}


def test_load_config_uses_cache_for_same_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "cached.yml"
    config_path.write_text("name: cached\n", encoding="utf-8")
    original_safe_load = yaml.safe_load
    safe_load_calls = 0

    def counting_safe_load(stream):
        nonlocal safe_load_calls
        safe_load_calls += 1
        return original_safe_load(stream)

    ConfigLoader.load_config.cache_clear()
    monkeypatch.setattr("finance_modeling.config.config.yaml.safe_load", counting_safe_load)

    first = ConfigLoader.load_config(str(config_path))
    second = ConfigLoader.load_config(str(config_path))

    assert first == second == {"name": "cached"}
    assert safe_load_calls == 1


def test_load_model_config_parses_models_from_project_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "experiment_config.yml").write_text(
        "- experiment_name: custom_experiment\n"
        "- name: GARCH\n"
        "  hyperparameters_list:\n"
        "    - p: 1\n"
        "      q: 1\n"
        "- name: PSOQRNN\n"
        "  hyperparameters_list:\n"
        "    - window_size: 10\n",
        encoding="utf-8",
    )
    ConfigLoader.load_config.cache_clear()
    monkeypatch.setattr("finance_modeling.config.config.get_main_root", lambda: str(tmp_path))

    config = ConfigLoader().load_model_config()

    assert config.experiment_name == "custom_experiment"
    assert [model.name for model in config.models] == ["GARCH", "PSOQRNN"]
    assert config.models[0].hyperparameters_list == [{"p": 1, "q": 1}]
    assert config.models[1].hyperparameters_list == [{"window_size": 10}]


def test_load_data_config_parses_assets_from_project_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "data_loading_config.yml").write_text(
        "- symbol: BTC-USD\n"
        "  asset_type: crypto\n"
        "  data_folder: bitcoin\n"
        "  description: Bitcoin\n"
        "  active: true\n",
        encoding="utf-8",
    )
    ConfigLoader.load_config.cache_clear()
    monkeypatch.setattr("finance_modeling.config.config.get_main_root", lambda: str(tmp_path))

    data_config = ConfigLoader().load_data_config()

    assert len(data_config.assets) == 1
    assert data_config.assets[0].symbol == "BTC-USD"
    assert data_config.assets[0].asset_type.value == "crypto"
    assert data_config.assets[0].active is True
