# Trabajo Final de Maestría: Modelos de Volatilidad Financiera

Proyecto en Python para comparar modelos de volatilidad financiera sobre series temporales, con un pipeline modular de carga de datos, preprocesamiento, entrenamiento, predicción, evaluación y comparación estadística.

Estado actual resumido:
- baseline implementado: `GARCH`
- challengers implementados: `CEEMDAN-LSTM` y `PSO-QRNN`
- runner con persistencia de predicciones, evaluaciones y comparaciones
- suite de tests operativa con cobertura medida

## Objetivo

El objetivo del repositorio es ejecutar corridas experimentales reproducibles para comparar distintos modelos de volatilidad sobre activos financieros, usando una arquitectura separada en configuración, datos, modelos, evaluación y utilidades.

El flujo actual es:

```text
load -> preprocess -> fit -> predict -> evaluate -> compare -> persist
```

## Stack

- Python `>=3.11,<3.14`
- Poetry para dependencias y entorno
- `arch` para modelos GARCH
- `pmdarima` para `auto_arima`
- `emd-signal` / `PyEMD` para CEEMDAN
- `torch` y `torchvision` para modelos basados en PyTorch
- `pytest` y `pytest-cov` para pruebas y cobertura

## Estructura del proyecto

```text
src/finance_modeling/
	config/        # carga de configuracion YAML
	data/          # loaders y preprocessors
	evaluation/    # metricas, evaluacion y comparacion
	experiments/   # runner principal
	models/        # GARCH, CEEMDAN-LSTM, PSO-QRNN y base comun
	schemas/       # modelos Pydantic y contratos de datos
	utils/         # helpers, logger y excepciones
tests/           # suite pytest
```

## Modelos disponibles

### GARCH

Baseline actual del proyecto. Usa residuales de `auto_arima` y selección de hiperparámetros por AIC.

### CEEMDAN-LSTM

Challenger basado en descomposición CEEMDAN y entrenamiento de una red LSTM por IMF, con validación interna para selección de hiperparámetros.

### PSO-QRNN

Challenger basado en una red recurrente con pérdida cuantílica y selección de hiperparámetros mediante PSO sobre el espacio definido en configuración.

## Instalación

### 1. Instalar Poetry

Si aún no lo tienes:

```powershell
pip install poetry
```

### 2. Instalar dependencias del proyecto

Desde la raíz del repositorio:

```powershell
poetry install
```

Este proyecto ya está configurado para instalar `torch` y `torchvision` desde el índice CUDA definido en `pyproject.toml`.

### 3. Verificar PyTorch / CUDA

```powershell
poetry run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

## Configuración

### Configuración de modelos

Archivo: [src/finance_modeling/config/experiment_config.yml](src/finance_modeling/config/experiment_config.yml)

Define:
- nombre del experimento
- modelos a ejecutar
- listas de hiperparámetros por modelo
- parámetros de PSO para `PSOQRNN`

### Configuración de datos

Archivo: [src/finance_modeling/config/data_loading_config.yml](src/finance_modeling/config/data_loading_config.yml)

Define:
- activos activos/inactivos
- tipo de activo
- carpeta de datos
- columna a usar

## Ejecutar el pipeline

La corrida principal se hace con el runner:

```powershell
poetry run python src/finance_modeling/experiments/runner.py
```

El runner hace lo siguiente por cada activo activo:
- carga el CSV curado
- preprocesa la serie
- divide train/test
- entrena cada modelo configurado
- genera predicciones
- calcula `RMSE` y `MAE`
- compara cada challenger contra `GARCH`
- persiste artefactos en disco

## Artefactos generados

Los resultados se guardan en:

```text
src/finance_modeling/results/models/<experiment_name>/
```

Actualmente el pipeline persiste, por modelo y activo:
- modelo serializado: `*.pkl`
- predicciones tabulares: `*_results.csv`
- mejores hiperparámetros: `*_best_hyperparameters.json`
- evaluación individual: `*_evaluation.json`
- comparación baseline vs challenger: `*_comparison.json`

Ejemplos de archivos esperados:
- `GARCH_BTC-USD.pkl`
- `GARCH_BTC-USD_results.csv`
- `GARCH_BTC-USD_evaluation.json`
- `GARCH_vs_PSO-QRNN_BTC-USD_comparison.json`

## Tests

La suite actual usa `pytest`.

Ejecutar todos los tests:

```powershell
poetry run python -m pytest tests
```

Ejecutar un archivo específico:

```powershell
poetry run python -m pytest tests/test_garch.py
```

## Cobertura de código

Generar cobertura en consola:

```powershell
poetry run python -m pytest tests --cov=src/finance_modeling --cov-report=term-missing
```

Generar artefactos persistentes:

```powershell
poetry run python -m pytest tests --cov=src/finance_modeling --cov-report=xml --cov-report=html
```

Artefactos generados:
- [coverage.xml](coverage.xml)
- [htmlcov](htmlcov)

Estado medido al 2026-04-19:
- `45` tests pasando
- cobertura global aproximada: `61%`

## Logging

El logger escribe en consola y en archivo dentro de:

```text
src/finance_modeling/logs/
```

Los modelos PyTorch reportan al inicio:
- versión de `torch`
- disponibilidad de CUDA
- dispositivo elegido (`cpu` o `cuda`)

## Limitaciones y advertencias actuales

- `README` operativo ya existe, pero la documentación metodológica aún no está completa.
- La configuración de datos avanzó, pero todavía hay inconsistencias entre algunos `symbol` y los nombres físicos de archivos curados.
- `Gold` aún no está incorporado en `data_loading_config.yml`.
- La cobertura actual protege bien contratos y pipeline, pero todavía es baja en internals costosos de `PSOQRNN` y `CEEMDAN-LSTM`.

## Próximos pasos recomendados

1. Corregir coherencia entre `data_loading_config.yml` y los archivos reales de Forex/índices.
2. Subir la cobertura global por encima de `70%`, empezando por `model_factory`, `PSOQRNN` y `CEEMDAN-LSTM`.
3. Ejecutar corridas oficiales comparativas multiactivo y consolidar tablas finales para el documento de maestría.

## Autor

Andres Molina
