/**
 * Transformación Heikin-Ashi (Fase 10b) — se calcula ACÁ, en el frontend,
 * a partir de las velas OHLCV que ya trae `/api/ohlcv` (el backend no
 * sabe nada de esto, ver el pedido explícito de la fase: "se calcula en
 * el frontend"). Fórmula estándar:
 *
 *     HA_close_t = (open_t + high_t + low_t + close_t) / 4
 *     HA_open_t  = (HA_open_{t-1} + HA_close_{t-1}) / 2
 *                  (primera vela: (open_0 + close_0) / 2, no hay previa)
 *     HA_high_t  = max(high_t, HA_open_t, HA_close_t)
 *     HA_low_t   = min(low_t, HA_open_t, HA_close_t)
 *
 * Es una transformación puramente VISUAL: suaviza el ruido vela a vela
 * para que las tendencias se vean más limpias, a costa de que el
 * open/high/low/close de cada vela Heikin-Ashi YA NO es el precio real
 * negociado en ese momento — por eso los overlays/osciladores (SMA, RSI,
 * VWAP, etc., calculados por el backend) siguen usando el precio REAL
 * tal cual devuelve `/api/studies`, nunca los valores Heikin-Ashi.
 */

import type { Candle } from "./types";

export function toHeikinAshi(candles: Candle[]): Candle[] {
  const result: Candle[] = [];
  let prevOpen: number | null = null;
  let prevClose: number | null = null;

  for (const candle of candles) {
    const haClose: number = (candle.open + candle.high + candle.low + candle.close) / 4;
    const haOpen: number =
      prevOpen !== null && prevClose !== null ? (prevOpen + prevClose) / 2 : (candle.open + candle.close) / 2;
    const haHigh = Math.max(candle.high, haOpen, haClose);
    const haLow = Math.min(candle.low, haOpen, haClose);

    result.push({ ...candle, open: haOpen, high: haHigh, low: haLow, close: haClose });
    prevOpen = haOpen;
    prevClose = haClose;
  }

  return result;
}
