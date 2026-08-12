# crypto-quant-desk

Herramienta cuantitativa de decisión de trading en cripto.

**Estado actual: Fase 0** — solo andamiaje del proyecto y capa de datos. Todavía
no hay modelos (GARCH, ARIMA, ML) implementados.

## Convenciones

- Python 3.10+, con `from __future__ import annotations` y type hints en todo el código.
- Retornos siempre en escala **decimal** (0.01 = 1%), nunca en porcentaje.
- Frecuencia de trabajo por defecto: **diaria** (`PERIODS_PER_YEAR = 252`).
- Todas las series de precios usan índice `DatetimeIndex` en **UTC**, tz-aware.
- Paths resueltos vía `pathlib` a partir de `config.BASE_DIR`, sin hardcodear rutas.

## Estructura

```
crypto-quant-desk/
├── config.py              # Config centralizada: paths, universo de activos, parámetros
├── data/
│   ├── loaders.py         # get_prices(): descarga/estandariza/cachea OHLCV
│   └── cache/              # Parquet cacheado (ignorado en git) + CSV de verificación (sí en git)
├── scripts/
│   └── build_dataset.py   # CLI para poblar la caché de todo el UNIVERSE
└── tests/
    └── test_loaders.py    # Tests offline (sin red) de la capa de datos
```

## Universo de activos

BTC, ETH, SOL, BNB, LTC (ver `config.UNIVERSE` para el mapeo de símbolos por fuente).

## Fuentes de datos

- **Binance** (`api.binance.com/api/v3/klines`): fuente primaria, OHLCV real, sin API key.
- **CoinMetrics** (CSV público en GitHub): fallback y verificación offline. Solo cierre
  (`PriceUSD`, o `ReferenceRateUSD` si el activo no publica `PriceUSD`); OHLC se rellena
  con el close y `volume` queda en NaN. Caveat conocido: para SOL puntualmente el CSV
  community de CoinMetrics solo trae ~7 días de `ReferenceRateUSD` (no histórico completo);
  para BTC/ETH/BNB/LTC sí cubre el histórico diario completo.
- **CoinGecko** (`market_chart/range`, público): segundo fallback. Solo cierre + volumen
  agregado. El plan gratuito limita el histórico a ~365 días, y el endpoint viene exigiendo
  cada vez más una API key (puede responder 401 sin ella).

## Uso

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Descargar y cachear el histórico de todo el universo (además exporta CSV de
verificación de CoinMetrics a `data/cache/`):

```bash
python scripts/build_dataset.py
```

Correr los tests (offline, no requieren red):

```bash
pytest
```
