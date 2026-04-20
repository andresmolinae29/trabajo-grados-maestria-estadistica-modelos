# TODO - Trabajo de Grado: Modelos de Volatilidad Financiera

## Estado actual real

- [x] Infraestructura modular separada en `config`, `data`, `models`, `evaluation`, `experiments`, `schemas` y `utils`.
- [x] Pipeline minimo operativo: `load -> preprocess -> fit -> predict -> evaluate -> persist`.
- [ ] Cobertura experimental todavia incompleta: la configuracion de datos ya incluye varios activos, pero sigue inconsistente con algunos archivos reales y aun no incluye Gold.
- [ ] Cierre metodologico todavia incompleto por README vacio, cobertura parcial de codigo y falta de corrida comparativa oficial con artefactos finales para varios activos.

## Corte tecnico (2026-04-19)

- [x] `runner.py` orquesta carga, preprocesamiento, entrenamiento, prediccion, evaluacion y guardado de artefactos.
- [x] `GARCHModel` implementado con baseline efectivo sobre residuales de `auto_arima` y seleccion de hiperparametros por AIC.
- [x] `CEEMDANLSTMModel` implementado con descomposicion CEEMDAN, tuning por validacion y entrenamiento por IMF.
- [x] `PSOQRNNModel` implementado con entrenamiento QRNN, prediccion cuantílica y seleccion de hiperparametros por PSO sobre el espacio configurado.
- [x] Persistencia de modelos, predicciones, hiperparametros y evaluaciones ya integrada al runner.
- [x] Capa de evaluacion disponible con RMSE, MAE y comparador con Diebold-Mariano.
- [x] Comparacion entre modelos integrada y persistida desde el runner mediante `ComparisonResult`.
- [x] Cobertura de tests ya iniciada y consolidada con suite `pytest` operativa.
- [x] Exportacion tabular de `PredictionResult` corregida a CSV por fila.
- [x] Logging de dispositivo CPU/GPU agregado en modelos PyTorch al iniciar.
- [ ] `README.md` sigue vacio y sin instrucciones operativas.
- [ ] Cobertura profunda aun pendiente en internals costosos de `CEEMDANLSTMModel` y `PSOQRNNModel`.

## Cobertura de codigo (2026-04-19)

- [x] Suite `pytest` agregada al proyecto y ejecutandose desde Poetry/.venv.
- [x] Suite actual consolidada: `45 passed`.
- [x] Cobertura global medida con `pytest-cov`: `61%` sobre `src/finance_modeling`.
- [x] Artefactos de cobertura generados: `coverage.xml` y `htmlcov/`.
- [x] Cobertura alta en `config`, `data`, `evaluation`, `utils`, `runner` y contratos base de modelos.
- [ ] Cobertura insuficiente aun en internals de `model_factory`, `CEEMDANLSTMModel` y `PSOQRNNModel`.

## Fase 1: Endurecer baseline y corrida actual

- [x] Dejar `GARCHModel` como baseline formal y estable para el experimento.
- [x] Verificar que el flujo `load -> preprocess -> fit -> predict -> evaluate` funcione desde el runner.
- [x] Confirmar que `PredictionResult` y `EvaluationResult` son los artefactos canonicos del pipeline.
- [x] Integrar persistencia de resultados y evaluaciones en `results/models/`.
- [x] Hacer explicita en codigo la decision metodologica actual: baseline `auto_arima + GARCH`.
- [x] Verificar que los CSV de predicciones se exporten en formato tabular por fila y no como serializacion cruda del objeto `PredictionResult`.
- [x] Agregar una verificacion de entorno para distinguir claramente ejecucion CPU vs GPU al iniciar modelos PyTorch.

## Fase 2: Completar cobertura de configuracion experimental

- [ ] Agregar a `data_loading_config.yml` el activo pendiente ya presente en `data/files/`: Gold.
- [ ] Confirmar para cada activo la coherencia entre `symbol`, `data_folder`, archivo fisico y `column_to_use`.
- [ ] Corregir especificamente la configuracion de EUR/USD, S&P500 y NASDAQ para alinear `symbol` y ruta real con los archivos curados existentes.
- [ ] Revisar si la frecuencia fija `15min` del loader aplica realmente a todos los datasets cargados.
- [x] Estandarizar nombres de modelos en configuracion segun las keys reales del registry.
- [ ] Decidir si la activacion de modelos en `experiment_config.yml` debe quedar limitada a baseline + un challenger por corrida para controlar tiempo de ejecucion.

## Fase 3: Cerrar el primer challenger utilizable

- [x] Implementar un primer challenger funcional: `CEEMDANLSTMModel`.
- [x] Implementar un segundo challenger funcional: `PSOQRNNModel`.
- [ ] Validar estabilidad numerica y costo computacional de `CEEMDANLSTMModel` sobre al menos un activo completo.
- [ ] Revisar la estrategia de validacion interna de `CEEMDANLSTMModel` para evitar fugas o sobrecostos innecesarios al descomponer y entrenar por cada combinacion.
- [x] Corregir o confirmar la exportacion de artefactos del challenger para que queden comparables con el baseline.
- [ ] Validar estabilidad numerica y costo computacional de `PSOQRNNModel` sobre al menos un activo completo.
- [ ] Profundizar tests de seleccion PSO, scoring interno y forecast autoregresivo de `PSOQRNNModel`.

## Fase 4: Ejecutar comparacion comparativa real

- [x] Consolidar `Evaluator` y `ModelComparator` como capa oficial de evaluacion.
- [x] Calcular RMSE y MAE de forma uniforme para todos los modelos implementados.
- [ ] Ejecutar baseline vs `CEEMDAN-LSTM` sobre el mismo activo y horizonte para producir un `ComparisonResult` real.
- [x] Persistir resultados comparativos por activo/modelo y no solo evaluaciones individuales.
- [ ] Definir formato de salida para tablas comparativas finales por activo y por modelo.
- [ ] Usar Diebold-Mariano solo cuando las predicciones baseline/challenger esten perfectamente alineadas en timestamps y horizonte.

## Fase 5: Reproducibilidad y calidad minima

- [x] Agregar tests unitarios para `ConfigLoader`, `Metrics`, `Evaluator`, `RawDataLoader` y `DataPreprocessor`.
- [x] Agregar al menos un smoke test del runner con GARCH sobre un dataset pequeno controlado.
- [x] Cubrir con test el contrato minimo de cualquier modelo: `fit`, `predict`, `PredictionResult`, persistencia basica.
- [x] Cubrir con tests los contratos principales de `GARCHModel`, `PSOQRNNModel` y `CEEMDANLSTMModel`.
- [ ] Agregar tests especificos para `ModelFactory` y validar errores de registry.
- [ ] Subir la cobertura global por encima de 70% con foco en `model_factory`, `PSOQRNNModel` y `CEEMDANLSTMModel`.
- [ ] Revisar logging y manejo de errores para fallos de carga, fitting y prediccion.
- [ ] Documentar en `README.md` como instalar dependencias, correr el pipeline y donde quedan los artefactos.

## Fase 6: Expansion metodologica

- [ ] Estabilizar y comparar `PSOQRNNModel` despues de tener baseline y `CEEMDANLSTMModel` validados en corridas oficiales.
- [ ] Evaluar si conviene unificar configuracion de experimento y datos en un solo archivo mas adelante.
- [ ] Organizar resultados finales para analisis, tablas y redaccion del documento de maestria.
- [ ] Preparar una matriz final de comparacion por modelo, activo y metrica.

## Riesgos y brechas visibles

- [ ] Evitar que el runner crezca con logica especifica por modelo al integrar comparaciones o controles de hardware.
- [ ] Evitar que metricas o comparaciones vuelvan a quedar dentro de las clases de modelo.
- [ ] Confirmar que la serializacion de predicciones y modelos siga siendo usable al crecer el numero de activos y challengers.
- [ ] Mantener separadas las responsabilidades de configuracion, datos, modelos, evaluacion y reporte.
- [ ] Evitar sobreestimar la calidad metodologica por tener tests de contrato: aun faltan pruebas sobre internals numericos y corridas oficiales multi-activo.
