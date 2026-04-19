---
name: exec-arch
description: 'Revisa avance tecnico e infraestructura de proyectos de modelado financiero en Python y actualiza TODO.md con estado real, brechas y prioridades. Usar cuando se pida auditoria de progreso, health-check arquitectonico o replanificacion del roadmap.'
argument-hint: 'Que foco quieres revisar (baseline, evaluacion, datos, tests, docs) y con que profundidad?'
---

# Exec Arch

## Objetivo

Convertir una peticion de "revisar avance e infraestructura" en una auditoria ejecutable del repositorio, con actualizacion accionable de `TODO.md` basada en evidencia del codigo.

## Cuando usar

- Cuando el usuario pide revisar progreso real del proyecto vs roadmap.
- Cuando se necesita validar si la arquitectura sigue separacion de responsabilidades.
- Cuando hace falta actualizar `TODO.md` para reflejar estado actual y proximos pasos.

## Procedimiento

1. Definir alcance de la auditoria
- Identificar si el pedido es workspace-scoped (proyecto actual) o personal.
- Confirmar foco principal: pipeline baseline, cobertura de modelos, evaluacion, tests, documentacion, o todos.

2. Levantar evidencia de infraestructura
- Revisar estructura de carpetas y modulos nucleares (`config`, `data`, `models`, `evaluation`, `experiments`, `schemas`, `utils`).
- Inspeccionar archivos de orquestacion y contratos (`runner`, factory, clases base, schemas, configs).
- Verificar coherencia entre configuraciones y registry de modelos.

3. Medir avance funcional
- Validar pipeline minimo: `load -> preprocess -> fit -> predict -> evaluate`.
- Clasificar cada componente en estado: implementado, parcial, stub, o pendiente.
- Confirmar artefactos canonicos de salida (predicciones, evaluacion, comparacion) y si se persisten.

4. Aplicar logica de decision
- Si el baseline esta implementado pero challengers estan en stub: priorizar hardening + reproducibilidad antes de expansion.
- Si hay evaluacion de metricas pero sin comparacion ejecutada: marcar Diebold-Mariano y tablas comparativas como siguiente hito.
- Si faltan tests o README operativo: elevar como deuda tecnica critica de cierre metodologico.
- Si configuracion de activos/modelos no cubre objetivos del experimento: priorizar completitud de config antes de tuning.

5. Actualizar `TODO.md` con evidencia
- Marcar solo tareas claramente implementadas como completadas.
- Agregar corte tecnico con fecha cuando ayude a trazabilidad.
- Mantener wording accionable y verificable (evitar tareas ambiguas).

6. Cerrar con chequeo de calidad
- Confirmar que no se sobreestimo el avance.
- Confirmar que los proximos pasos son secuenciales y ejecutables.
- Confirmar que el TODO refleja tanto progreso como riesgos visibles.

## Criterios de completitud

- Existe una lectura explicita del estado de infraestructura del repo.
- `TODO.md` queda actualizado con checks consistentes con el codigo.
- Se explicitan brechas bloqueantes (tests, docs, challengers, persistencia, cobertura de activos).
- Se propone una priorizacion clara del siguiente ciclo de trabajo.

## Entregable esperado

- Actualizacion de `TODO.md` con estado real del proyecto.
- Resumen corto de hallazgos arquitectonicos.
- Lista de ambiguedades abiertas para que el usuario confirme decisiones metodologicas.
