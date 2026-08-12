# CryptoQuant — Dossier de Proyecto (v0.1)

> Documento fundacional. Es un **borrador vivo**: lo iteramos antes de escribir código.
> Objetivo de este archivo: fijar alcance, arquitectura y roadmap para trabajar prolijo.

---

## 1. Visión

Construir una **herramienta de decisión de trading en cripto**, de nivel profesional,
que integre análisis cuantitativo (econometría + machine learning) y **gestión de riesgo**
para responder, ante cada activo y momento, una pregunta operativa concreta:

> **¿Conviene operar ahora, en qué dirección y con qué tamaño de posición?**

No es un oráculo de precios. Es un **copiloto de decisión** que combina evidencia y expone
su nivel de confianza, con el riesgo siempre a la vista.

---

## 2. Qué ES y qué NO ES (alcance honesto)

**ES:**
- Un motor de decisión que integra régimen de volatilidad, señales y sizing por riesgo.
- Dos modos de operación: **direccional** (comprar/vender un activo) y **pares / stat-arb** (market-neutral).
- Un backtester riguroso con costos, para no engañarnos.
- Una UI de escritorio tipo terminal.

**NO ES:**
- Un predictor mágico del precio exacto (ARIMA sobre precio de cripto ≈ random walk; se usa solo como baseline).
- **Arbitraje entre exchanges** (eso es guerra de latencia / infraestructura; no es viable para retail).
- **HFT / intradía de minutos** (comisiones y microestructura lo vuelven poco realista para empezar).
- Asesoramiento financiero. Es una herramienta de análisis. Cripto es de altísimo riesgo.

---

## 3. Principios de diseño

1. **Reutilizar, no reinventar.** Partimos de los `modules/` del repo `momentum`
   (GARCH t-Student + Kupiec, VaR/ES, Sharpe de Lo, Markowitz). Cambio principal: **mensual → diario**
   (`periods_per_year`: 12 → 252).
2. **Decisión, no predicción de precio.** El objetivo es la calidad de la decisión de trade, no el R² sobre el precio.
3. **El riesgo primero.** Sizing por volatilidad (vol targeting) y límites de riesgo son parte del núcleo, no un extra.
4. **Validación como control de calidad.** Walk-forward con costos, purging + embargo. Sin esto no confiamos en nada.
5. **Config-driven.** Parámetros, universo, costos y fechas centralizados (estilo `config.py` del repo actual).

---

## 4. Universo de activos

| Activo | Rol |
|---|---|
| BTC | Ancla del mercado / direccional / pata de pares |
| ETH | Direccional / pata de pares (BTC–ETH es el par cointegrado clásico) |
| SOL | Direccional / candidato de pares |
| BNB | Direccional / candidato de pares |
| LTC | Candidato de pares (historia larga, cointegración con BTC) — opcional |

Los pares no se fijan a dedo: se **testean por cointegración** y solo quedan los que pasan.

---

## 5. El producto: el motor de decisión

Flujo de un solo pipeline (no modelos compitiendo):

```
   Datos (OHLCV diario)
        │
        ▼
 [1] Régimen de volatilidad   ──►  GARCH / EGARCH / GJR  (¿calma o tormenta?)
        │
        ▼
 [2] Señal primaria           ──►  Indicadores (RSI/MACD/Bollinger/ATR)
        │                          + spread de pares (modo stat-arb)
        ▼
 [3] Filtro + sizing (ML)      ──►  Triple-barrier + Gradient Boosting + meta-labeling
        │                          (¿esta señal vale la pena? ¿cuánto apostar?)
        ▼
 [4] Overlay de riesgo         ──►  vol targeting, VaR/ES, límites
        │
        ▼
   RECOMENDACIÓN: acción + confianza + tamaño sugerido + racional por componente
```

**Núcleo predictivo elegido:** triple-barrier + gradient boosting con **meta-labeling**.
Se eligió porque su salida *es* la decisión operativa ("operar / no operar + tamaño"),
en lugar de un pronóstico de precio. El modelo primario define el **lado**; el meta-modelo
define el **tamaño** filtrando señales de baja calidad.

---

## 6. Los dos modos de trading

### A) Direccional (por activo)
Comprar/vender BTC, ETH, etc. según el score del motor. Es la "calculadora que dice si conviene operar".

### B) Pares / Stat-arb (módulo aparte, market-neutral)
Si dos activos están cointegrados y el spread se estira, se va **largo del barato y corto del caro**,
apostando a la reversión. Gana aunque el mercado entero caiga.
Componentes: Engle-Granger + Johansen + ADF sobre el spread, **half-life** de reversión,
**hedge ratio dinámico (filtro de Kalman)** vs. OLS estático, z-score de entrada/salida, sizing por Kelly.

Ambos modos comparten backtester y métricas de riesgo. En la UI, "tipo de operación" es un selector.

---

## 7. Metodología (resumen)

- **Volatilidad:** GARCH(1,1) t-Student (base del repo) + **EGARCH/GJR** para capturar asimetría
  (las caídas disparan más vol que las subidas). Selección por AIC/BIC.
- **Baseline direccional:** ARIMA — honesto y de bajo peso; sirve de referencia, no de predictor principal.
- **Etiquetado ML:** **triple-barrier** (take-profit / stop-loss / tiempo, escalados por volatilidad local).
- **Sizing ML:** **meta-labeling** (modelo primario = lado; meta-modelo = tamaño).
- **Validación:** **purged K-fold + embargo** (el K-fold estándar filtra información en finanzas).
- **Evaluación anti-overfitting:** walk-forward con costos, **deflated Sharpe** / combinatorial purged CV.
- **Librería de apoyo:** `mlfinlab` / `mlfinpy` (implementaciones de López de Prado) para no reescribir todo.

---

## 8. Stack técnico

- **Lenguaje:** Python 3.10+
- **Cuant:** pandas, numpy, scipy, statsmodels (ARIMA), arch (GARCH), scikit-learn, XGBoost/LightGBM
- **Fin-ML:** mlfinlab/mlfinpy
- **Datos:** Binance klines (primaria, paginada), CoinGecko / CoinMetrics (fallback)
- **UI:** **PySide6** (Qt de escritorio, tipo terminal) — con threading para no congelar la UI en cómputos largos
- **Repo:** GitHub (flujo: vos pusheás → yo clono y verifico → devuelvo feedback + próximos prompts)

> Nota de verificación: desde el entorno de Claude no hay acceso a Binance/CoinGecko (solo a GitHub).
> Para que Claude pueda verificar, se commitea un CSV histórico (CoinMetrics) en el repo.

---

## 9. Datos (estado 2026)

| Fuente | Uso | Límite relevante |
|---|---|---|
| Binance klines | Primaria (diario, backtest) | Sin API key; ~1200 req/min; ~500 velas/request (paginar) |
| CoinGecko | Fallback | Histórico gratis limitado a ~365 días |
| CoinMetrics | Fallback + verificación | Histórico diario completo y gratis vía GitHub |

---

## 10. Roadmap por fases

| Fase | Nombre | Entregable | Estado |
|---|---|---|---|
| 0 | Andamiaje | Repo, `config.py`, `data/loaders.py`, CSV commiteado, tests base | ☐ |
| 1 | Motor clásico | Indicadores + ARIMA baseline + GARCH (EGARCH/GJR) + engine + **backtester con costos** | ☐ |
| 2 | Stat-arb / pares | Cointegración + Kalman + z-score + Kelly | ☐ |
| 3 | Capa ML | Triple-barrier + XGBoost + meta-labeling + validación purged | ☐ |
| 4 | UI | Terminal PySide6 (gráficos, semáforo, tablas, sizing) | ☐ |
| 5 | Opcional | DL experimental + detección de régimen (HMM) | ☐ |

Cada fase entrega algo usable por sí sola. Al final de la Fase 1 ya tenés una "calculadora" operativa.

---

## 11. Riesgos y advertencias

- **Overfitting de backtest:** el enemigo número uno. Se combate con purging/embargo y deflated Sharpe.
- **Data leakage:** mirar el futuro sin querer. Auditar cada feature.
- **Costos y slippage:** un backtest sin costos miente. Van desde el día uno.
- **Régimen cambiante:** cripto cambia de régimen brusco; ningún modelo entrenado es permanente.
- **Disclaimer:** herramienta de análisis/educación, no asesoramiento financiero. Riesgo alto de pérdida.

---

## 12. Decisiones abiertas / próximos pasos

- [ ] Confirmar UI: **PySide6** (propuesto) vs PyQt5 vs Streamlit.
- [ ] Confirmar LTC dentro o fuera del universo de pares.
- [ ] Definir nombre del repo/proyecto.
- [ ] **Arrancar Fase 0**: primer prompt para Claude Code (andamiaje + loaders + config).

---

*v0.1 — borrador para iterar. Autor del proyecto: MLEIV. Asistencia de diseño: Claude.*
