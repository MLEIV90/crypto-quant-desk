/**
 * Tipos TypeScript que reflejan los esquemas Pydantic de la API
 * (`api/models.py`, Fase 8a) — un tipo por respuesta que consume este
 * frontend. Fase 8b solo necesita `/api/assets`, `/api/ohlcv` y
 * `/api/studies`; el resto de los endpoints (`/api/suggester`, `/api/risk`,
 * `/api/backtest`, `/api/garch-series`) se tipan cuando se consuman en 8c.
 *
 * Convención: fechas llegan como string ISO 8601 UTC (p. ej.
 * "2026-08-12T00:00:00Z") — se parsean recién donde hace falta un
 * timestamp numérico (ver `components/Chart.tsx`), nunca acá.
 */

export interface AssetsResponse {
  activos: string[];
  timeframes: string[];
}

export interface Candle {
  fecha: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

export interface OHLCVResponse {
  asset: string;
  interval: string;
  velas: Candle[];
}

export interface SoporteResistencia {
  resistencia: number[];
  soporte: number[];
  precio_actual: number;
}

export interface StudiesResponse {
  asset: string;
  interval: string;
  fechas: string[];
  sma_20: (number | null)[];
  sma_50: (number | null)[];
  ema_12: (number | null)[];
  ema_26: (number | null)[];
  bb_upper: (number | null)[];
  bb_mid: (number | null)[];
  bb_lower: (number | null)[];
  rsi_14: (number | null)[];
  macd: (number | null)[];
  macd_signal: (number | null)[];
  macd_hist: (number | null)[];
  stoch_k: (number | null)[];
  stoch_d: (number | null)[];
  fibonacci: Record<string, number> | null;
  soporte_resistencia: SoporteResistencia;
  pivotes: Record<string, number>;
}
