"""Terminal de escritorio (PySide6) de crypto-quant-desk.

REGLA DE ARQUITECTURA — separación estricta modelo/vista: la UI (`app.main`,
`app.widgets`) SOLO dispara cálculos y muestra resultados; nunca calcula
nada por su cuenta. Todo el cómputo pesado (ajustar GARCH, etc.) corre en un
`QThread` (`app.workers.AnalysisWorker`) para no congelar la interfaz, y
vuelve a la ventana principal por señales/slots de Qt — el patrón estándar
para no bloquear el hilo de eventos de Qt con trabajo largo. Ningún módulo
de acá reimplementa cálculos: todos llaman al backend ya existente
(`data.loaders`, `models.garch`, `metrics.risk_measures`, `signals.engine`,
`signals.returns`).
"""
