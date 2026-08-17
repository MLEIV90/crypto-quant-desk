/**
 * Reglas de alerta y su evaluación (Fase 8d) — lógica pura, sin React,
 * separada de `components/AlertsPanel.tsx` a propósito para que sea fácil
 * de leer/testear en aislado.
 *
 * Evaluación 100% client-side sobre los datos que ya trae `/api/studies`
 * (Fase 8a) — sin backend nuevo. "Cruza" se detecta comparando las
 * ÚLTIMAS DOS velas de la serie correspondiente (no un valor estático):
 * la condición debe haberse cumplido justo en la vela más reciente.
 */

import type { OHLCVResponse, StudiesResponse } from "./types";

export type AlertRuleType =
  | "rsi_above"
  | "rsi_below"
  | "price_cross_sma20_up"
  | "price_cross_sma20_down"
  | "price_touch_sr"
  | "macd_cross_signal_up"
  | "macd_cross_signal_down";

export interface AlertRule {
  id: string;
  asset: string;
  interval: string;
  type: AlertRuleType;
  threshold: number | null;
  createdAt: string;
}

export const RULE_LABELS: Record<AlertRuleType, string> = {
  rsi_above: "RSI cruza por encima de",
  rsi_below: "RSI cruza por debajo de",
  price_cross_sma20_up: "Precio cruza SMA20 hacia arriba",
  price_cross_sma20_down: "Precio cruza SMA20 hacia abajo",
  price_touch_sr: "Precio toca un soporte/resistencia",
  macd_cross_signal_up: "MACD cruza su señal hacia arriba",
  macd_cross_signal_down: "MACD cruza su señal hacia abajo",
};

export const RULE_TYPES_WITH_THRESHOLD: AlertRuleType[] = ["rsi_above", "rsi_below"];

export const DEFAULT_THRESHOLD: Partial<Record<AlertRuleType, number>> = {
  rsi_above: 70,
  rsi_below: 30,
};

export function describeRule(rule: AlertRule): string {
  const label = RULE_LABELS[rule.type];
  return RULE_TYPES_WITH_THRESHOLD.includes(rule.type) && rule.threshold !== null
    ? `${label} ${rule.threshold}`
    : label;
}

function lastTwo(values: (number | null)[]): { prev: number; curr: number } | null {
  const n = values.length;
  if (n < 2) return null;
  const prev = values[n - 2];
  const curr = values[n - 1];
  if (prev === null || curr === null) return null;
  return { prev, curr };
}

export function evaluateRule(rule: AlertRule, studies: StudiesResponse, ohlcv: OHLCVResponse): boolean {
  switch (rule.type) {
    case "rsi_above": {
      const pair = lastTwo(studies.rsi_14);
      if (!pair || rule.threshold === null) return false;
      return pair.prev <= rule.threshold && pair.curr > rule.threshold;
    }
    case "rsi_below": {
      const pair = lastTwo(studies.rsi_14);
      if (!pair || rule.threshold === null) return false;
      return pair.prev >= rule.threshold && pair.curr < rule.threshold;
    }
    case "price_cross_sma20_up":
    case "price_cross_sma20_down": {
      const smaPair = lastTwo(studies.sma_20);
      const closes = ohlcv.velas;
      if (!smaPair || closes.length < 2) return false;
      const prevClose = closes[closes.length - 2].close;
      const currClose = closes[closes.length - 1].close;
      if (rule.type === "price_cross_sma20_up") {
        return prevClose <= smaPair.prev && currClose > smaPair.curr;
      }
      return prevClose >= smaPair.prev && currClose < smaPair.curr;
    }
    case "price_touch_sr": {
      const lastCandle = ohlcv.velas[ohlcv.velas.length - 1];
      if (!lastCandle) return false;
      const levels = [...studies.soporte_resistencia.resistencia, ...studies.soporte_resistencia.soporte];
      const tolerance = lastCandle.close * 0.003; // 0.3% de tolerancia alrededor del nivel
      return levels.some(
        (level) => Math.abs(lastCandle.high - level) <= tolerance || Math.abs(lastCandle.low - level) <= tolerance,
      );
    }
    case "macd_cross_signal_up":
    case "macd_cross_signal_down": {
      const macdPair = lastTwo(studies.macd);
      const signalPair = lastTwo(studies.macd_signal);
      if (!macdPair || !signalPair) return false;
      if (rule.type === "macd_cross_signal_up") {
        return macdPair.prev <= signalPair.prev && macdPair.curr > signalPair.curr;
      }
      return macdPair.prev >= signalPair.prev && macdPair.curr < signalPair.curr;
    }
    default:
      return false;
  }
}
