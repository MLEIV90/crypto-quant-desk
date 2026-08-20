/**
 * Reglas de alerta y su evaluación (Fase 8d, ampliado en 13c) — lógica
 * pura, sin React, separada de `components/AlertsPanel.tsx` a propósito
 * para que sea fácil de leer/testear en aislado.
 *
 * Evaluación 100% client-side sobre los datos que ya trae `/api/studies`/
 * `/api/ohlcv`/`/api/volume-profile` (sin backend nuevo). "Cruza"/"sale de"
 * se detecta comparando las ÚLTIMAS DOS velas de la serie correspondiente
 * (no un valor estático): la condición debe haberse cumplido justo en la
 * vela más reciente, no en cualquier momento del pasado.
 *
 * HONESTIDAD (Fase 13c, sin cambios de fondo respecto de 8d): estas
 * alertas son TÉCNICAS, no predicen nada — son solo el aviso de que una
 * condición que EL USUARIO eligió (un cruce de precio, un nivel de RSI,
 * etc.) se cumplió en los datos. Ver `ALERTS_HONESTY_HELP` en
 * `./helpTexts.ts`.
 *
 * LIMITACIÓN ARQUITECTURAL (Fase 13c): una regla para un activo distinto
 * al que está cargado en pantalla (`studies`/`ohlcv` de `/api/studies` son
 * de UN activo a la vez) no se puede evaluar hasta que el usuario cambie a
 * ver ESE activo — no hay polling en background de los 5 activos a la vez.
 * Por eso las reglas se pueden CREAR para cualquier moneda (Fase 13c), pero
 * solo se EVALÚAN en vivo mientras esa moneda está cargada en pantalla
 * (ver `AlertsPanel.tsx`, que marca cuáles están "en vivo" ahora mismo).
 */

import type { OHLCVResponse, StudiesResponse, VolumeProfileResponse } from "./types";

export type MaKey = "sma_20" | "sma_50" | "ema_12" | "ema_26";

export type AlertRuleType =
  | "rsi_above"
  | "rsi_below"
  | "stoch_above"
  | "stoch_below"
  | "price_cross_level_up"
  | "price_cross_level_down"
  | "price_cross_ma_up"
  | "price_cross_ma_down"
  | "price_touch_sr"
  | "price_touch_poc"
  | "price_touch_value_area"
  | "macd_cross_signal_up"
  | "macd_cross_signal_down"
  | "bollinger_break_upper"
  | "bollinger_break_lower";

export interface AlertRule {
  id: string;
  asset: string;
  interval: string;
  type: AlertRuleType;
  /** Umbral de RSI/Estocástico, o nivel de precio para price_cross_level_*. null si no aplica. */
  threshold: number | null;
  /** Media a usar para price_cross_ma_*. null si no aplica. */
  maKey: MaKey | null;
  /** Toggle sin borrar (Fase 13c) — una regla deshabilitada nunca se evalúa. */
  enabled: boolean;
  createdAt: string;
}

export interface AlertHistoryEntry {
  id: string;
  ruleId: string;
  asset: string;
  interval: string;
  message: string;
  timestamp: string;
}

export const RULE_LABELS: Record<AlertRuleType, string> = {
  rsi_above: "RSI cruza por encima de",
  rsi_below: "RSI cruza por debajo de",
  stoch_above: "Estocástico (%K) cruza por encima de",
  stoch_below: "Estocástico (%K) cruza por debajo de",
  price_cross_level_up: "Precio cruza el nivel hacia arriba",
  price_cross_level_down: "Precio cruza el nivel hacia abajo",
  price_cross_ma_up: "Precio cruza una media hacia arriba",
  price_cross_ma_down: "Precio cruza una media hacia abajo",
  price_touch_sr: "Precio toca un soporte/resistencia",
  price_touch_poc: "Precio toca el POC (Volume Profile)",
  price_touch_value_area: "Precio toca el borde del Value Area",
  macd_cross_signal_up: "MACD cruza su señal hacia arriba",
  macd_cross_signal_down: "MACD cruza su señal hacia abajo",
  bollinger_break_upper: "Precio sale de la banda de Bollinger superior",
  bollinger_break_lower: "Precio sale de la banda de Bollinger inferior",
};

export const RULE_TYPES_WITH_THRESHOLD: AlertRuleType[] = [
  "rsi_above",
  "rsi_below",
  "stoch_above",
  "stoch_below",
  "price_cross_level_up",
  "price_cross_level_down",
];

export const RULE_TYPES_WITH_LEVEL_INPUT: AlertRuleType[] = ["price_cross_level_up", "price_cross_level_down"];

export const RULE_TYPES_WITH_MA: AlertRuleType[] = ["price_cross_ma_up", "price_cross_ma_down"];

export const DEFAULT_THRESHOLD: Partial<Record<AlertRuleType, number>> = {
  rsi_above: 70,
  rsi_below: 30,
  stoch_above: 80,
  stoch_below: 20,
};

export const MA_LABELS: Record<MaKey, string> = {
  sma_20: "SMA 20",
  sma_50: "SMA 50",
  ema_12: "EMA 12",
  ema_26: "EMA 26",
};

const MA_FIELD: Record<MaKey, "sma_20" | "sma_50" | "ema_12" | "ema_26"> = {
  sma_20: "sma_20",
  sma_50: "sma_50",
  ema_12: "ema_12",
  ema_26: "ema_26",
};

export function describeRule(rule: AlertRule): string {
  let label = RULE_LABELS[rule.type];
  if (RULE_TYPES_WITH_MA.includes(rule.type) && rule.maKey) {
    const maLabel = MA_LABELS[rule.maKey];
    label = rule.type === "price_cross_ma_up" ? `Precio cruza ${maLabel} hacia arriba` : `Precio cruza ${maLabel} hacia abajo`;
  } else if (RULE_TYPES_WITH_THRESHOLD.includes(rule.type) && rule.threshold !== null) {
    label = `${label} ${rule.threshold}`;
  }
  return `${rule.asset} (${rule.interval}) — ${label}`;
}

function lastTwo(values: (number | null)[]): { prev: number; curr: number } | null {
  const n = values.length;
  if (n < 2) return null;
  const prev = values[n - 2];
  const curr = values[n - 1];
  if (prev === null || curr === null) return null;
  return { prev, curr };
}

function lastTwoCloses(ohlcv: OHLCVResponse): { prev: number; curr: number } | null {
  const velas = ohlcv.velas;
  if (velas.length < 2) return null;
  return { prev: velas[velas.length - 2].close, curr: velas[velas.length - 1].close };
}

export function evaluateRule(
  rule: AlertRule,
  studies: StudiesResponse,
  ohlcv: OHLCVResponse,
  volumeProfile?: VolumeProfileResponse | null,
): boolean {
  if (!rule.enabled) return false;

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
    case "stoch_above": {
      const pair = lastTwo(studies.stoch_k);
      if (!pair || rule.threshold === null) return false;
      return pair.prev <= rule.threshold && pair.curr > rule.threshold;
    }
    case "stoch_below": {
      const pair = lastTwo(studies.stoch_k);
      if (!pair || rule.threshold === null) return false;
      return pair.prev >= rule.threshold && pair.curr < rule.threshold;
    }
    case "price_cross_level_up": {
      const closes = lastTwoCloses(ohlcv);
      if (!closes || rule.threshold === null) return false;
      return closes.prev <= rule.threshold && closes.curr > rule.threshold;
    }
    case "price_cross_level_down": {
      const closes = lastTwoCloses(ohlcv);
      if (!closes || rule.threshold === null) return false;
      return closes.prev >= rule.threshold && closes.curr < rule.threshold;
    }
    case "price_cross_ma_up":
    case "price_cross_ma_down": {
      if (!rule.maKey) return false;
      const maPair = lastTwo(studies[MA_FIELD[rule.maKey]]);
      const closes = lastTwoCloses(ohlcv);
      if (!maPair || !closes) return false;
      if (rule.type === "price_cross_ma_up") {
        return closes.prev <= maPair.prev && closes.curr > maPair.curr;
      }
      return closes.prev >= maPair.prev && closes.curr < maPair.curr;
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
    case "price_touch_poc": {
      const lastCandle = ohlcv.velas[ohlcv.velas.length - 1];
      if (!lastCandle || !volumeProfile) return false;
      const tolerance = lastCandle.close * 0.003;
      return (
        Math.abs(lastCandle.high - volumeProfile.poc) <= tolerance ||
        Math.abs(lastCandle.low - volumeProfile.poc) <= tolerance
      );
    }
    case "price_touch_value_area": {
      const lastCandle = ohlcv.velas[ohlcv.velas.length - 1];
      if (!lastCandle || !volumeProfile) return false;
      const tolerance = lastCandle.close * 0.003;
      const levels = [volumeProfile.value_area_low, volumeProfile.value_area_high];
      return levels.some(
        (level) => Math.abs(lastCandle.high - level) <= tolerance || Math.abs(lastCandle.low - level) <= tolerance,
      );
    }
    case "macd_cross_signal_up": {
      const macdPair = lastTwo(studies.macd);
      const signalPair = lastTwo(studies.macd_signal);
      if (!macdPair || !signalPair) return false;
      return macdPair.prev <= signalPair.prev && macdPair.curr > signalPair.curr;
    }
    case "macd_cross_signal_down": {
      const macdPair = lastTwo(studies.macd);
      const signalPair = lastTwo(studies.macd_signal);
      if (!macdPair || !signalPair) return false;
      return macdPair.prev >= signalPair.prev && macdPair.curr < signalPair.curr;
    }
    case "bollinger_break_upper": {
      const bandPair = lastTwo(studies.bb_upper);
      const closes = lastTwoCloses(ohlcv);
      if (!bandPair || !closes) return false;
      return closes.prev <= bandPair.prev && closes.curr > bandPair.curr;
    }
    case "bollinger_break_lower": {
      const bandPair = lastTwo(studies.bb_lower);
      const closes = lastTwoCloses(ohlcv);
      if (!bandPair || !closes) return false;
      return closes.prev >= bandPair.prev && closes.curr < bandPair.curr;
    }
    default:
      return false;
  }
}
