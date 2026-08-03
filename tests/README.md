# Tests

La carpeta `tests/` espeja los subpaquetes de `src/finance_modeling/`:

```
tests/
├── factories.py          # helpers compartidos entre subpaquetes (make_asset_metadata)
├── config/                test_config.py         -> finance_modeling.config
├── data/                  test_data.py            -> finance_modeling.data
├── evaluation/             test_evaluation.py      -> finance_modeling.evaluation.evaluator / metrics
│                          test_comparison.py       -> finance_modeling.evaluation.comparison
├── experiments/            test_runner.py           -> finance_modeling.experiments.runner
├── models/                factories.py             # helper compartido make_series
│                          test_models_base.py      -> finance_modeling.models.base
│                          test_garch.py            -> finance_modeling.models.garch
│                          test_ceemdan_lstm.py     -> finance_modeling.models.ceemdan_lstm
│                          test_psoqrnn.py          -> finance_modeling.models.psoqrnn
│                          test_model_factory.py    -> finance_modeling.models.model_factory
│                          test_utils.py            -> finance_modeling.models.utils
└── utils/                 test_utils.py            -> finance_modeling.utils
```

`finance_modeling.schemas` no tiene archivo de test dedicado (son modelos pydantic declarativos, cubiertos indirectamente por el resto de la suite).

## Correr los tests

```bash
pytest                        # suite completa, coverage se calcula por defecto (ver pyproject.toml)
pytest -m "not stale_api"     # excluye la deuda técnica documentada abajo
pytest tests/models           # solo un subpaquete
```

La configuración de pytest/coverage vive en `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.coverage.run]`, `[tool.coverage.report]`).

## Coverage actual (98% total, `src/finance_modeling`)

| Módulo | Cobertura | Nota |
|---|---|---|
| `models/base.py`, `models/psoqrnn.py`, `models/model_factory.py`, `models/utils.py` | 100% | — |
| `models/ceemdan_lstm.py` | 96% | descomposición CEEMDAN real (`ceemdan_features`) mockeada por ser costosa; línea de logging cada 10 epochs sin cubrir |
| `runner.py` | 98% | solo falta el guard `if __name__ == "__main__"` |
| `garch.py` | 94% | ramas de error de convergencia sin cubrir |
| `evaluator.py` / `comparison.py` / `preprocessors.py` / `config.py` / `utils/common.py` | 92-96% | ramas de error/edge case puntuales sin cubrir |
| resto de módulos | 100% | — |

## Deuda técnica conocida

**3 tests marcados `@pytest.mark.skip` + `@pytest.mark.stale_api`** en `test_runner.py`: prueban `runner.enrich_prediction_result`, una función que ya no existe en la implementación actual de `runner.py` (el pipeline se reescribió con `build_prediction_result`/`save_prediction_results_to_csv`/`save_forecast_plot`). No hay equivalente directo que rescribir — quedan documentados como deuda, filtrables con `pytest -m stale_api`.

Los otros 12 tests `stale_api` (en `test_ceemdan_lstm.py`, `test_psoqrnn.py`, `test_models_base.py`) fueron reescritos contra la API actual de cada modelo (constructor con `symbol=`, `fit()`/`predict()` sin argumentos). Para mantener los tests rápidos y deterministas: `GARCHModel` mockea `arch.arch_model`; `PSOQRNNModel` mockea `PSO.optimize` (además de un test liviano con `n_particles`/`max_iter` reducidos que sí corre el PSO real); `CEEMDANLSTMModel` mockea `ceemdan_features` (la descomposición CEEMDAN real es costosa) y reduce `WINDOW_SIZE`/`BATCH_SIZE`/`EPOCHS` vía `monkeypatch.setattr` sobre los atributos de clase.

**Bug de `predicted_volatility` corregido** (`src/finance_modeling/evaluation/comparison.py`, `evaluator.py`): `PredictionRow` solo expone `predicted_value`; `predicted_volatility` era únicamente un alias de *entrada* (`AliasChoices`), no un atributo legible después de construir el objeto. Ambos módulos hacían `row.predicted_volatility` en tiempo de lectura, lo cual siempre lanzaba `AttributeError`. Se corrigió usando directamente `row.predicted_value`.
