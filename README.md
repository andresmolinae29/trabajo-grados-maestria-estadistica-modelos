# Trabajo Final de Maestría: Modelos de Volatilidad Financiera

Código del trabajo final de la Maestría en Ciencias - Estadística *Comparación de metodologías estadísticas y de aprendizaje profundo para el modelamiento de la volatilidad en series financieras intradía*.

El proyecto compara tres metodologías de pronóstico de volatilidad, la familia GARCH, CEEMDAN-LSTM y PSOQRNN, entre sí y frente a un pronóstico ingenuo. Usa seis activos con precios cada 15 minutos entre el 5 de febrero y el 2 de julio de 2026. La variable objetivo es la volatilidad realizada: la desviación estándar móvil de los retornos en una ventana de 252 barras, escalada por $\sqrt{252}$.

Estado al 2026-11-10:
- experimento oficial de la tesis: `20261104_causal`, generado con `experiment_config_causal.yml`, con resultados en `src/finance_modeling/results/models/20261104_causal/`
- `experiment_config.yml` y su experimento `20261031` conservan la especificación de los artículos originales, sin la adaptación causal
- modelos: `NAIVE`, `GARCH` (GARCH, GJR-GARCH y FIGARCH), `CEEMDAN_LSTM` y `PSOQRNN`
- pruebas: 126, todas pasan; cobertura del 98 %
- documento de la tesis en `proyecto/`, en corrección tras la evaluación del jurado

## Flujo

```text
load -> preprocess -> [tune] -> fit -> predict -> evaluate -> compare -> persist
```

El ajuste de hiperparámetros (`tune`) solo aplica a las dos redes. La evaluación y la comparación se hacen sobre las fechas que cubren todos los modelos.

## Stack

- Python `>=3.11,<3.14`
- Poetry para dependencias y entorno
- `arch` para la familia GARCH y `pmdarima` (`auto_arima`) para el modelo de media
- `emd-signal` (`PyEMD`) para CEEMDAN
- `torch` para las redes LSTM y QRNN
- `statsmodels` y `scipy` para el diagnóstico de los modelos GARCH
- `dieboldmariano` para la prueba de Diebold-Mariano
- `plotly` y `kaleido` para las figuras de la tesis
- `pytest` y `pytest-cov` para pruebas y cobertura

## Datos

Los CSV curados están en `src/finance_modeling/data/files/<carpeta>/<símbolo>_curated.csv`. Se obtuvieron de la API de Yahoo Finance con el módulo de extracción del repositorio [trabajo-grados-maestria-estadistica-obtencion-datos](https://github.com/andresmolinae29/trabajo-grados-maestria-estadistica-obtencion-datos).

| Símbolo | Carpeta | Clase |
|---|---|---|
| `BTC-USD` | `bitcoin` | Criptomoneda |
| `ETH-USD` | `ethereum` | Criptomoneda |
| `EURUSD=X` | `eur_usd` | Divisa |
| `GC=F` | `gold` | Materia prima |
| `^GSPC` | `sp500` | Índice bursátil |
| `^IXIC` | `nasdaq` | Índice bursátil |

## Estructura del proyecto

```text
src/finance_modeling/
    config/        # configuración YAML: experimento, datos y preselección GARCH
    data/          # carga de CSV y preprocesamiento (retornos, innovaciones, volatilidad realizada)
    evaluation/    # métricas, evaluación, comparación y diagnóstico GARCH
    experiments/   # runner, ajuste de las redes y script de diagnóstico GARCH
    models/        # NAIVE, GARCH, CEEMDAN-LSTM, PSOQRNN y base común
    schemas/       # modelos Pydantic y contratos de datos
    utils/         # helpers, logger y excepciones
    results/       # artefactos de cada experimento, diagnósticos GARCH y caché del CEEMDAN causal
tests/             # suite pytest, espejo de src/finance_modeling
proyecto/          # tesis en LaTeX, bibliografía, figuras y scripts de tablas y figuras
```

## Modelos

### NAIVE

Pronóstico ingenuo: repite el último valor observado de la volatilidad realizada. Es la referencia mínima que debe superar cualquier modelo.

### GARCH

- **Modelo de media:** un ARIMA ajustado con `auto_arima`. Sus residuos se estandarizan a varianza unitaria antes de estimar la varianza condicional.
- **Candidatos:** trece, definidos en `experiment_config.yml`. Combinan GARCH, GJR-GARCH y FIGARCH con distribución normal, $t$ de Student o $t$ asimétrica.
- **Selección:** se preseleccionan por activo en `garch_selection.yml` a partir del diagnóstico de entrenamiento, y entre ellos decide el AIC.
- **Pronóstico:** a un paso. Para que mida lo mismo que la variable objetivo, se agrega sobre la misma ventana de 252 barras.

### CEEMDAN-LSTM

- **Entrada:** los retornos se descomponen con CEEMDAN, con los parámetros por defecto de PyEMD. Cada instante se representa con todas las componentes (IMF y residuo) y la volatilidad realizada del mismo instante.
- **Descomposición causal (`causal: true`):** en cada barra se descompone la ventana de los 512 retornos más recientes, con un máximo de seis IMF más el residuo, y se conserva el último valor de cada componente. Se guarda en caché en `results/causal_ceemdan/<símbolo>.npz`. Sin esta opción, como en Cao et al. (2019), se descompone la serie completa, que incluye observaciones futuras.
- **Contexto de entrenamiento (`train_context: true`):** las primeras secuencias de prueba toman como contexto las últimas filas de entrenamiento, para pronosticar desde la primera barra de prueba.
- **Red:** una única red LSTM predice la volatilidad del período siguiente. La configuración de partida tiene dos capas de 128 unidades y una capa densa de 64.
- **Ajuste:** sin búsqueda, la ventana $W$ y las épocas $E$ se eligen por validación entre seis combinaciones. En la configuración oficial se agregan 30 combinaciones aleatorias de una grilla de diez hiperparámetros (`search_space`). Se evalúan tres semillas.

### PSOQRNN

- **Red:** una QRNN *feedforward* de una capa oculta, entrenada con pérdida cuantílica en la mediana.
- **Entrenamiento:** sus pesos se buscan con PSO. Los valores de partida son los de Pradeepkumar y Ravi (2017): 60 partículas, 5 000 iteraciones, $w=0,8$, $c_1=c_2=2$ y velocidad en $[-5, 5]$.
- **Entradas:** la volatilidad realizada del período anterior y una innovación del retorno. Con `causal: true` es $\varepsilon_{t-1}$, centrada con la media de entrenamiento. Sin esa opción, como en el artículo, es la innovación contemporánea $\varepsilon_t$, que contiene el retorno que entra en el valor pronosticado.
- **Ajuste:** sin búsqueda, el número de neuronas ocultas ($2$ a $5$) se elige por validación. En la configuración oficial se agregan 60 combinaciones aleatorias de una grilla de ocho hiperparámetros: neuronas, partículas, iteraciones, $w$, $c_1$, $c_2$, $v_{\max}$ y rango de inicialización. Se evalúan tres semillas.

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

Todos los archivos están en `src/finance_modeling/config/`.

- **[experiment_config.yml](src/finance_modeling/config/experiment_config.yml):** nombre del experimento (que también es la carpeta de salida), modelos a ejecutar, lista de hiperparámetros candidatos de cada modelo y semillas de las redes. Sigue la especificación de los artículos (experimento `20261031`).
- **[experiment_config_causal.yml](src/finance_modeling/config/experiment_config_causal.yml):** configuración oficial de la tesis (experimento `20261104_causal`). Tiene los mismos candidatos GARCH y activa, en las redes, `causal`, `train_context` (solo CEEMDAN-LSTM) y la búsqueda aleatoria (`search_space` y `n_trials`).
- **[data_loading_config.yml](src/finance_modeling/config/data_loading_config.yml):** activos, cuáles están activos, carpeta de datos, clase de activo y columna de precio.
- **[garch_selection.yml](src/finance_modeling/config/garch_selection.yml):** modelos GARCH preseleccionados por activo, con su justificación. Un activo que no aparezca aquí se selecciona por AIC entre todos los candidatos.

## Ejecución

### Corrida completa

Configuración oficial de la tesis:

```powershell
poetry run python src/finance_modeling/experiments/runner.py --config experiment_config_causal.yml
```

Sin `--config` se usa `experiment_config.yml`. Con `--symbol` (repetible) se limita la corrida a ciertos activos. En Windows, `poetry run` elimina el `^` de `^GSPC` y `^IXIC`; para esos símbolos hay que usar `.venv\Scripts\python.exe` directamente.

La descomposición causal es costosa, pues requiere una descomposición por barra. Puede calcularse antes y dejarse en caché:

```powershell
poetry run python src/finance_modeling/experiments/run_causal_ceemdan.py --symbol BTC-USD
```

Por cada activo activo, el runner:
- carga el CSV curado y construye retornos, innovaciones y volatilidad realizada
- divide la serie 80/20 en entrenamiento y prueba
- en las redes, elige hiperparámetros y semilla por validación (64 % entrenamiento, 16 % validación)
- ajusta cada modelo, guarda el modelo y sus predicciones, y en el GARCH su diagnóstico
- en las redes, repite el ajuste con las otras semillas para medir la variabilidad
- calcula RMSE y MAE sobre las fechas comunes a todos los modelos
- compara cada modelo con `NAIVE` y con `GARCH` mediante la prueba de Diebold-Mariano

### Diagnóstico GARCH de un activo

```powershell
poetry run python src/finance_modeling/experiments/run_garch_diagnostics.py --symbol BTC-USD
```

Escribe en `src/finance_modeling/results/garch_diagnostics/<símbolo>/`:
- estadística de los retornos y FAC
- pruebas previas a la estimación
- evaluación de los candidatos con las reglas de admisibilidad
- coeficientes del modelo elegido
- análisis de los residuos estandarizados

### Tablas de la tesis

```powershell
poetry run python proyecto/build_results_summary.py --experiment 20261104_causal
```

Sin `--experiment` se usa `20261031`.

Genera en la carpeta del experimento:
- `summary_metrics.csv`: MAE, RMSE, razón frente al ingenuo, sesgo, correlación y otros indicadores
- `summary_comparisons.csv`: prueba de Diebold-Mariano con varianza robusta (Newey-West)
- `summary_networks.csv`: hiperparámetros elegidos y variabilidad entre semillas

### Figuras de la tesis

```powershell
poetry run python proyecto/generar_figuras.py --solo modelos --outdir proyecto/las_figuras/20261104_causal
```

- Genera las figuras de precios (`iniciales/`) y las de volatilidad predicha frente a realizada de cada modelo y activo.
- Con `--solo precios` o `--solo modelos` se genera solo una de las dos familias.
- `--experimento` indica la carpeta de resultados; por defecto, `20261104_causal`.
- La tesis toma las figuras de modelos de `proyecto/las_figuras/20261104_causal/`, y las de precios y diagnóstico GARCH de `proyecto/las_figuras/20261031/`, porque no cambiaron con la corrida causal.
- Las figuras de diagnóstico GARCH (`diagnostics/`) no las genera este script: son copias renombradas de las que produce `run_garch_diagnostics.py` en `src/finance_modeling/results/garch_diagnostics/<símbolo>/`.

## Artefactos generados

Cada experimento se guarda en `src/finance_modeling/results/models/<experiment_name>/`. Por modelo y activo:
- `<MODELO>_<símbolo>_predictions.csv`: predicción y valor real por fecha
- `<MODELO>_<símbolo>_evaluation.json`: RMSE y MAE sobre las fechas comunes
- `<MODELO>_<símbolo>_forecast.png`: figura de control
- `<modelo>_<símbolo>.pkl`: modelo serializado
- `<REFERENCIA>_vs_<MODELO>_<símbolo>_comparison.json`: comparación con `NAIVE` o `GARCH`; la prueba de Diebold-Mariano usa aquí varianza simple ($h=1$)
- solo en las redes, `<MODELO>_<símbolo>_tuning.csv` (resultados de validación) y `<MODELO>_<símbolo>_seeds.csv` (error de prueba por semilla)
- `garch_diagnostics/<símbolo>/`: diagnóstico del modelo GARCH ajustado

La descomposición CEEMDAN causal no depende del experimento y se guarda aparte, en `src/finance_modeling/results/causal_ceemdan/<símbolo>.npz`.

## Documento de la tesis

El documento está en `proyecto/proyecto.tex` y la bibliografía en `proyecto/mi_bibliografia.bib`. El estilo bibliográfico es `proyecto/apalike-es.bst`, una copia de `apalike` con las palabras fijas en español («y», «En», «pp.», «edición»). Para compilarlo con `latexmk` desde `proyecto/`, BibTeX necesita saber dónde están el `.bib` y el `.bst` cuando la salida va a otra carpeta:

```powershell
cd proyecto
$env:BIBINPUTS = "$PWD;"
$env:BSTINPUTS = "$PWD;"
latexmk -pdf -synctex=1 -outdir=build/proyecto proyecto.tex
```

## Tests

La suite usa `pytest`, y la cobertura se calcula por defecto (configuración en `pyproject.toml`).

Ejecutar todos los tests:

```powershell
poetry run python -m pytest tests
```

Ejecutar un archivo específico:

```powershell
poetry run python -m pytest tests/models/test_garch.py
```

Generar los reportes de cobertura `coverage.xml` y `htmlcov/`:

```powershell
poetry run python -m pytest tests --cov-report=xml --cov-report=html
```

Estado medido al 2026-11-10: las 126 pruebas pasan y la cobertura global es del 98 %. Quedan sin cubrir los bloques `if __name__ == "__main__"`, dos ramas de la descomposición causal en `ceemdan_lstm.py` (relleno cuando un corte produce menos IMF y cálculo con varios procesos) y la validación de longitud de los retornos en `psoqrnn.py`. La organización de la suite está en [tests/README.md](tests/README.md).

## Logging

El logger escribe en consola y en archivo dentro de:

```text
src/finance_modeling/logs/
```

Los modelos PyTorch reportan al inicio:
- versión de `torch`
- disponibilidad de CUDA
- dispositivo elegido (`cpu` o `cuda`)

## Limitaciones

Las limitaciones del estudio están en la Sección 7.1 de la tesis. Las que afectan la lectura de los resultados de este código son:
- **Adaptación causal:** la configuración oficial evita el sesgo de anticipación de los artículos, pero la ventana de 512 retornos y el máximo de seis IMF se fijaron de antemano y no se evaluó su sensibilidad. `experiment_config.yml` (`20261031`) sigue los artículos y no es causal.
- **Búsqueda de hiperparámetros:** se evaluaron 36 configuraciones del CEEMDAN-LSTM y 64 del PSOQRNN, de grillas con más de 300 000 combinaciones, con un único período de validación.
- **Variable objetivo:** dos valores consecutivos comparten 251 de 252 retornos, lo que favorece al pronóstico ingenuo.
- **Período de prueba:** abarca cerca de un mes, sobre una única partición temporal.

## Licencia

MIT. Ver [LICENSE](LICENSE).

## Autor

Andres Molina
