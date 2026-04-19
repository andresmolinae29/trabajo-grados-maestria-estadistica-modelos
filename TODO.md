# TODO - Trabajo de Grado: Modelos de Volatilidad Financiera

## Estado actual estimado

- [ ] Consolidar el proyecto desde el ~45% de avance actual hacia una version experimental reproducible.
- [ ] Mantener el foco primero en cerrar el baseline y despues en ampliar la comparacion.

## Corte tecnico (2026-04-19)

- [x] Estructura modular base operativa: `config`, `data`, `models`, `evaluation`, `experiments`, `schemas`, `utils`.
- [x] Runner funcional para baseline sobre activos activos en configuracion.
- [x] `GARCHModel` implementado con seleccion de hiperparametros por AIC.
- [ ] Persistencia de artefactos de salida aun no integrada al runner.
- [ ] Challengers (`PSOQRNN`, `CEEMDAN-LSTM`) aun en estado skeleton.
- [ ] Cobertura de tests aun no iniciada (`tests/` solo con `__init__.py`).
- [ ] `README.md` aun sin documentacion operativa.

## Fase 1: Cerrar baseline GARCH

- [x] Dejar `GARCHModel` como baseline formal y estable para el experimento.
- [x] Verificar que el flujo `load -> preprocess -> fit -> predict -> evaluate` funcione sin ajustes manuales.
- [x] Confirmar que `PredictionResult` y `EvaluationResult` sean los artefactos canonicos del pipeline.
- [ ] Revisar persistencia de resultados en `results/models/` para guardar metricas y predicciones.
- [ ] Definir si el baseline usara solo GARCH o ARIMA + GARCH como decision metodologica explicita.

## Fase 2: Completar cobertura experimental

- [ ] Agregar a `data_loading_config.yml` los activos pendientes: ETH, EUR/USD, S&P500, NASDAQ y Gold.
- [ ] Confirmar que cada activo tenga ruta, carpeta y columna objetivo consistentes.
- [ ] Revisar si la frecuencia y el esquema de particion train/test son validos para todos los activos.
- [x] Estandarizar nombres de modelos en configuracion segun las keys reales del registry.

## Fase 3: Implementar el primer challenger real

- [ ] Implementar `PSOQRNNModel` como primer challenger despues de cerrar GARCH.
- [ ] Alinear el contrato de `PSOQRNNModel` con `BaseVolatilityModel`.
- [ ] Definir insumos adicionales del modelo: ventanas, tensores, semillas y salida esperada.
- [ ] Ejecutar comparacion baseline vs challenger sobre al menos un activo antes de escalar al resto.

## Fase 4: Completar evaluacion comparativa

- [x] Consolidar `Evaluator` y `ModelComparator` como capa oficial de evaluacion.
- [x] Calcular RMSE y MAE de forma uniforme para todos los modelos.
- [ ] Ejecutar test de Diebold-Mariano cuando ya existan baseline y challenger sobre el mismo horizonte.
- [ ] Definir formato de salida para tablas comparativas por activo y por modelo.

## Fase 5: Reproducibilidad y calidad minima

- [ ] Agregar tests unitarios para `ConfigLoader`, `Metrics`, `Evaluator` y `RawDataLoader`.
- [ ] Agregar al menos un smoke test del runner con GARCH.
- [ ] Revisar logging y manejo de errores para fallos de carga, fitting y prediccion.
- [ ] Documentar en `README.md` como correr el pipeline, que configuraciones usa y que artefactos genera.

## Fase 6: Expansion del trabajo de tesis

- [ ] Implementar `CEEMDANLSTMModel` una vez que GARCH y PSOQRNN esten estables.
- [ ] Evaluar si conviene unificar configuracion de experimento y datos en un solo archivo mas adelante.
- [ ] Organizar resultados finales para analisis, tablas y redaccion del documento de maestria.
- [ ] Preparar una matriz final de comparacion por modelo, activo y metrica.

## Riesgos y pendientes visibles

- [ ] Evitar que el runner crezca con logica especifica por modelo.
- [ ] Evitar que metricas o comparaciones vuelvan a quedar dentro de las clases de modelo.
- [ ] Mantener separadas las responsabilidades de configuracion, datos, modelos, evaluacion y reporte.
