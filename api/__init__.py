"""API REST (FastAPI) de crypto-quant-desk — Fase 8a.

Capa FINA sobre el backend ya existente: cada endpoint (`api/main.py`) arma
su respuesta llamando directo a funciones de `data.loaders`,
`signals.studies`, `signals.suggester`, `models.garch`,
`metrics.risk_measures` y `backtest.engine`, y las serializa a JSON con los
esquemas Pydantic de `api/models.py` — no reimplementa ningún cálculo.

Corre en paralelo a la terminal de escritorio (`app/`, PySide6): son dos
consumidores independientes del mismo backend, pensados para un futuro
frontend web — uno no depende del otro (`api/` no importa nada de `app/`).
"""
