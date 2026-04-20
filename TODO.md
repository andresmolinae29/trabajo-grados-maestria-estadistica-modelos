# TODO - Trabajo de Grado: Modelos de Volatilidad Financiera

## Estado actual real

- [x] Infraestructura modular separada en `config`, `data`, `models`, `evaluation`, `experiments`, `schemas` y `utils`.
- [x] Pipeline minimo operativo: `load -> preprocess -> fit -> predict -> evaluate -> persist`.
- [ ] Cobertura experimental todavia incompleta: solo BTC esta activo en configuracion de datos.
- [ ] Cierre metodologico incompleto por falta de tests, comparacion ejecutada y README operativo.

## Corte tecnico (2026-04-19)

- [x] `runner.py` orquesta carga, preprocesamiento, entrenamiento, prediccion, evaluacion y guardado de artefactos.
- [x] `GARCHModel` implementado con baseline efectivo sobre residuales de `auto_arima` y seleccion de hiperparametros por AIC.
- [x] `CEEMDANLSTMModel` implementado con descomposicion CEEMDAN, tuning por validacion y entrenamiento por IMF.
- [ ] `PSOQRNNModel` sigue en stub y no cumple todavia el contrato funcional del pipeline.
- [x] Persistencia de modelos, predicciones, hiperparametros y evaluaciones ya integrada al runner.
- [x] Capa de evaluacion disponible con RMSE, MAE y comparador con Diebold-Mariano.
- [ ] Comparacion entre modelos aun no ejecutada ni persistida desde el runner.
- [ ] Cobertura de tests aun no iniciada (`tests/` solo con `__init__.py`).
- [ ] `README.md` sigue vacio y sin instrucciones operativas.

## Fase 1: Endurecer baseline y corrida actual

- [x] Dejar `GARCHModel` como baseline formal y estable para el experimento.
- [x] Verificar que el flujo `load -> preprocess -> fit -> predict -> evaluate` funcione desde el runner.
- [x] Confirmar que `PredictionResult` y `EvaluationResult` son los artefactos canonicos del pipeline.
- [x] Integrar persistencia de resultados y evaluaciones en `results/models/`.
- [x] Hacer explicita en codigo la decision metodologica actual: baseline `auto_arima + GARCH`.
- [ ] Verificar que los CSV de predicciones se exporten en formato tabular por fila y no como serializacion cruda del objeto `PredictionResult`.
- [ ] Agregar una verificacion de entorno para distinguir claramente ejecucion CPU vs GPU al iniciar modelos PyTorch.

## Fase 2: Completar cobertura de configuracion experimental

- [ ] Agregar a `data_loading_config.yml` los activos pendientes ya presentes en `data/files/`: ETH, EUR/USD, S&P500, NASDAQ y Gold.
- [ ] Confirmar para cada activo la coherencia entre `symbol`, `data_folder`, archivo fisico y `column_to_use`.
- [ ] Revisar si la frecuencia fija `15min` del loader aplica realmente a todos los datasets cargados.
- [x] Estandarizar nombres de modelos en configuracion segun las keys reales del registry.
- [ ] Decidir si la activacion de modelos en `experiment_config.yml` debe quedar limitada a baseline + un challenger por corrida para controlar tiempo de ejecucion.

## Fase 3: Cerrar el primer challenger utilizable

- [x] Implementar un primer challenger funcional: `CEEMDANLSTMModel`.
- [ ] Validar estabilidad numerica y costo computacional de `CEEMDANLSTMModel` sobre al menos un activo completo.
- [ ] Revisar la estrategia de validacion interna de `CEEMDANLSTMModel` para evitar fugas o sobrecostos innecesarios al descomponer y entrenar por cada combinacion.
- [ ] Corregir o confirmar la exportacion de artefactos del challenger para que queden comparables con el baseline.
- [ ] Mantener `PSOQRNNModel` fuera de corridas oficiales hasta implementar `fit`, `predict` y contrato de salida.

## Fase 4: Ejecutar comparacion comparativa real

- [x] Consolidar `Evaluator` y `ModelComparator` como capa oficial de evaluacion.
- [x] Calcular RMSE y MAE de forma uniforme para todos los modelos implementados.
- [ ] Ejecutar baseline vs `CEEMDAN-LSTM` sobre el mismo activo y horizonte para producir un `ComparisonResult` real.
- [ ] Persistir resultados comparativos por activo/modelo y no solo evaluaciones individuales.
- [ ] Definir formato de salida para tablas comparativas finales por activo y por modelo.
- [ ] Usar Diebold-Mariano solo cuando las predicciones baseline/challenger esten perfectamente alineadas en timestamps y horizonte.

## Fase 5: Reproducibilidad y calidad minima

- [ ] Agregar tests unitarios para `ConfigLoader`, `Metrics`, `Evaluator`, `ModelFactory`, `RawDataLoader` y `DataPreprocessor`.
- [ ] Agregar al menos un smoke test del runner con GARCH sobre un dataset pequeno controlado.
- [ ] Cubrir con test el contrato minimo de cualquier modelo: `fit`, `predict`, `PredictionResult`, persistencia basica.
- [ ] Revisar logging y manejo de errores para fallos de carga, fitting y prediccion.
- [ ] Documentar en `README.md` como instalar dependencias, correr el pipeline y donde quedan los artefactos.

## Fase 6: Expansion metodologica

- [ ] Implementar `PSOQRNNModel` solo despues de tener baseline y `CEEMDANLSTMModel` estabilizados y comparados.
- [ ] Evaluar si conviene unificar configuracion de experimento y datos en un solo archivo mas adelante.
- [ ] Organizar resultados finales para analisis, tablas y redaccion del documento de maestria.
- [ ] Preparar una matriz final de comparacion por modelo, activo y metrica.

## Riesgos y brechas visibles

- [ ] Evitar que el runner crezca con logica especifica por modelo al integrar comparaciones o controles de hardware.
- [ ] Evitar que metricas o comparaciones vuelvan a quedar dentro de las clases de modelo.
- [ ] Confirmar que la serializacion de predicciones y modelos siga siendo usable al crecer el numero de activos y challengers.
- [ ] Mantener separadas las responsabilidades de configuracion, datos, modelos, evaluacion y reporte.
