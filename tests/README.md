# Tests

La carpeta `tests/` espeja los subpaquetes de `src/finance_modeling/`:

```
tests/
├── conftest.py           # fija el backend Agg de matplotlib para que las pruebas no abran ventanas
├── factories.py          # helpers compartidos entre subpaquetes (make_asset_metadata)
├── config/                test_config.py                -> finance_modeling.config
├── data/                  test_data.py                   -> finance_modeling.data
├── evaluation/             test_evaluation.py             -> finance_modeling.evaluation.evaluator / metrics
│                          test_comparison.py             -> finance_modeling.evaluation.comparison
│                          test_garch_diagnostics.py      -> finance_modeling.evaluation.garch_diagnostics
├── experiments/            test_runner.py                 -> finance_modeling.experiments.runner
│                          test_network_tuning.py         -> finance_modeling.experiments.network_tuning
│                          test_run_causal_ceemdan.py     -> finance_modeling.experiments.run_causal_ceemdan
│                          test_run_garch_diagnostics.py  -> finance_modeling.experiments.run_garch_diagnostics
├── models/                factories.py                   # helper compartido make_series
│                          test_models_base.py            -> finance_modeling.models.base
│                          test_naive.py                  -> finance_modeling.models.naive
│                          test_garch.py                  -> finance_modeling.models.garch
│                          test_ceemdan_lstm.py           -> finance_modeling.models.ceemdan_lstm
│                          test_psoqrnn.py                -> finance_modeling.models.psoqrnn
│                          test_model_factory.py          -> finance_modeling.models.model_factory
│                          test_utils.py                  -> finance_modeling.models.utils
└── utils/                 test_utils.py                  -> finance_modeling.utils
```

`finance_modeling.schemas` no tiene archivo de test dedicado (son modelos pydantic declarativos, cubiertos indirectamente por el resto de la suite).

## Correr los tests

```bash
pytest                        # suite completa, coverage se calcula por defecto (ver pyproject.toml)
pytest tests/models           # solo un subpaquete
pytest --cov-report=xml --cov-report=html   # además genera coverage.xml y htmlcov/
```

La configuración de pytest/coverage vive en `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.coverage.run]`, `[tool.coverage.report]`).

## Coverage actual (98% total, `src/finance_modeling`, medido el 2026-11-10)

Las 126 pruebas pasan; no hay pruebas omitidas.

| Módulo | Cobertura | Líneas sin cubrir |
|---|---|---|
| `experiments/run_causal_ceemdan.py` | 82% | bloque `if __name__ == "__main__"` (lectura de argumentos) |
| `experiments/run_garch_diagnostics.py` | 84% | bloque `if __name__ == "__main__"` (lectura de argumentos) |
| `models/ceemdan_lstm.py` | 94% | relleno con ceros cuando un corte produce menos IMF que el máximo, y cálculo de la descomposición causal con varios procesos (`Pool`) |
| `experiments/runner.py` | 95% | bloque `if __name__ == "__main__"` |
| `config/config.py` | 97% | bloque `if __name__ == "__main__"` |
| `models/psoqrnn.py` | 99% | error cuando los retornos no cubren la ventana de la volatilidad en el modo causal |
| resto de módulos | 100% | — |

## Cómo se mantienen rápidas

Las pruebas evitan los cálculos costosos reemplazándolos con dobles:
- `GARCHModel` reemplaza `arch.arch_model` y el modelo de media.
- `PSOQRNNModel` reemplaza `PSO.optimize`, salvo un test liviano con `n_particles`/`max_iter` reducidos que sí corre el PSO real.
- `CEEMDANLSTMModel` reemplaza `ceemdan_features`, porque la descomposición CEEMDAN real es costosa, y reduce `WINDOW_SIZE`/`BATCH_SIZE`/`EPOCHS` con `monkeypatch.setattr` sobre los atributos de clase. `ceemdan_features` se prueba aparte con un CEEMDAN falso.
- `causal_ceemdan_features` se prueba con un `last_components` falso, un solo proceso (`processes=1`) y la caché en una carpeta temporal. El modelo, la búsqueda de hiperparámetros y `run_causal_ceemdan.py` la reemplazan por una función falsa.
- El runner y los scripts de diagnóstico GARCH y de descomposición causal se prueban con modelos y cargadores falsos.
- La prueba de Diebold-Mariano de `ModelComparator.compare` sí usa la librería real `dieboldmariano`.

## Historial

- **2026-11-03 y 2026-11-04:** se agregaron pruebas para la variante causal y la búsqueda de hiperparámetros: que cada fila de la descomposición causal use solo la ventana que termina en esa barra y que la caché se reutilice; que el CEEMDAN-LSTM causal entrene solo con filas de descomposición completa y que `train_context` pronostique todas las filas de prueba; que el PSOQRNN causal reciba $\varepsilon_{t-1}$ centrada con la media de entrenamiento; que los candidatos conserven la grilla fija y agreguen combinaciones aleatorias distintas y reproducibles; que la búsqueda causal descomponga la serie una sola vez; y el script `run_causal_ceemdan.py`.
- **2026-11-02:** se eliminaron 3 pruebas marcadas `stale_api` en `test_runner.py`, que estaban omitidas porque probaban `runner.enrich_prediction_result`, una función que ya no existe. También se eliminaron sus funciones auxiliares (`FakeModel`, `FakePreprocessor`, `make_prediction`, `make_processed_data`) y el marcador `stale_api` de `pyproject.toml`. Se agregaron pruebas para lo que no estaba cubierto: `load_garch_selection`, el cálculo real de Diebold-Mariano, las ramas de error del GARCH y del preprocesamiento, `ceemdan_features`, el registro de pérdida por épocas, los candidatos GARCH que no se pueden ajustar, el diagnóstico solo para el GARCH en el runner, la continuidad del runner tras un error en un activo, y el script `run_garch_diagnostics.py`.
- **2026-11-02:** se eliminaron funciones de la API anterior que el pipeline ya no usaba, junto con sus pruebas y funciones auxiliares de prueba:
  - `Evaluator.from_timeinput_and_prediction_result` y `ModelComparator.compare_from_timeinput_and_prediction_results`.
  - En `utils/common.py`: `convert_list_to_series`, `generate_future_timestamps` y `prediction_results_to_dataframe`.
