"""Experimento de investigación (Fase 18): Deep Reinforcement Learning para
ASIGNACIÓN DE CARTERA sobre `config.UNIVERSE` (BTC/ETH/SOL/BNB/LTC) + efectivo.

HONESTIDAD ANTE TODO — léase antes de tocar cualquier resultado de este
paquete: esto es RESEARCH, con la misma vara anti-fugas y anti-overfitting
que el resto del proyecto (`ml/models.py`, `signals/suggester.py`). La
expectativa de base, y el resultado más probable, es que el agente NO le
gane de forma consistente a los baselines fuera de muestra — este proyecto
ya demostró varias veces (ML supervisado, indicadores técnicos, arbitraje
estadístico) que no hay un edge direccional fácil en estos datos. Si el
agente SÍ le gana a todo, el primer sospechoso es leakage o overfitting al
período de test, NO que "el RL encontró algo que el resto no encontró" —
tratá cualquier resultado positivo con más escepticismo, no menos.

Estructura del paquete:
- `rl.features`: dataset de observaciones (estacionarias, causales) y
  retornos, reutilizando `ml.features.build_feature_matrix` donde aplica.
- `rl.env`: `PortfolioEnv`, el entorno `gymnasium.Env` de asignación de
  cartera (acción continua -> softmax -> pesos que suman 1).
- `rl.baselines`: los 4 baselines de comparación (equiponderado, buy&hold
  BTC, vol targeting existente, asignador aleatorio).
- `rl.evaluation`: evaluación de una secuencia de pesos (retorno neto de
  costos + métricas, reutilizando `metrics.risk_measures`), partición
  walk-forward, y el orquestador del experimento completo
  (`run_walkforward_experiment`).

Punto de entrada para correr el experimento completo:
`scripts/run_rl_experiment.py`.

NO está conectado a ninguna vista del frontend ni a la API todavía, a
propósito (ver la Fase 18 del proyecto): el entrenamiento es lento y el
resultado esperado es nulo — primero se establece con evidencia propia si
esto aporta algo antes de exponerlo a un usuario final.
"""
