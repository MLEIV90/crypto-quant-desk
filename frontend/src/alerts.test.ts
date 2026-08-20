/**
 * Tests de `evaluateRule` (Fase 14, hallazgo F-02 de auditoría) — la
 * lógica CRÍTICA del panel de alertas: si esto se rompe, una alerta puede
 * no disparar nunca, dispararse siempre (spam), o dispararse en el momento
 * equivocado. No son tests de UI (eso es `AlertsPanel.tsx`, sin cubrir
 * acá) — solo la función pura `evaluateRule`/`describeRule` de `./alerts.ts`.
 */

import { describe, expect, it } from "vitest";
import { describeRule, evaluateRule, type AlertRule } from "./alerts";
import type { Candle, OHLCVResponse, StudiesResponse, VolumeProfileResponse } from "./types";

function makeStudies(overrides: Partial<StudiesResponse> = {}): StudiesResponse {
  const fechas = overrides.fechas ?? ["2021-01-01", "2021-01-02", "2021-01-03"];
  const n = fechas.length;
  const nulls = (): (number | null)[] => Array.from({ length: n }, () => null);

  return {
    asset: "BTC",
    interval: "1d",
    fechas,
    sma_20: nulls(),
    sma_50: nulls(),
    ema_12: nulls(),
    ema_26: nulls(),
    bb_upper: nulls(),
    bb_mid: nulls(),
    bb_lower: nulls(),
    rsi_14: nulls(),
    macd: nulls(),
    macd_signal: nulls(),
    macd_hist: nulls(),
    stoch_k: nulls(),
    stoch_d: nulls(),
    vwap: nulls(),
    obv: nulls(),
    ichimoku_tenkan: nulls(),
    ichimoku_kijun: nulls(),
    ichimoku_senkou_a: nulls(),
    ichimoku_senkou_b: nulls(),
    ichimoku_chikou: nulls(),
    fibonacci: null,
    soporte_resistencia: { resistencia: [], soporte: [], precio_actual: 0 },
    pivotes: {},
    ...overrides,
  };
}

function makeOhlcv(closes: number[], highs?: number[], lows?: number[]): OHLCVResponse {
  const velas: Candle[] = closes.map((close, i) => ({
    fecha: `2021-01-0${i + 1}`,
    open: close,
    high: highs ? highs[i] : close,
    low: lows ? lows[i] : close,
    close,
    volume: null,
  }));
  return { asset: "BTC", interval: "1d", velas };
}

function makeRule(overrides: Partial<AlertRule> = {}): AlertRule {
  return {
    id: "r1",
    asset: "BTC",
    interval: "1d",
    type: "rsi_above",
    threshold: 70,
    maKey: null,
    enabled: true,
    createdAt: "2021-01-01T00:00:00.000Z",
    ...overrides,
  };
}

describe("evaluateRule: reglas deshabilitadas", () => {
  it("nunca evalúa true, sin importar los datos", () => {
    const rule = makeRule({ type: "rsi_above", threshold: 50, enabled: false });
    const studies = makeStudies({ rsi_14: [40, 60] });
    const ohlcv = makeOhlcv([100, 101]);

    expect(evaluateRule(rule, studies, ohlcv)).toBe(false);
  });
});

describe("evaluateRule: rsi_above / rsi_below (cruce, no repetido)", () => {
  it("dispara SOLO en la vela donde el RSI cruza por encima del umbral", () => {
    const rule = makeRule({ type: "rsi_above", threshold: 70 });
    // RSI: 65 -> 75 (cruza en la 2da vela) -> 80 (ya venía arriba, no es cruce)
    const rsiSeries = [65, 75, 80];

    // Evaluando con datos hasta la vela del cruce (65 -> 75): true.
    const studiesAtCross = makeStudies({ rsi_14: rsiSeries.slice(0, 2), fechas: ["2021-01-01", "2021-01-02"] });
    expect(evaluateRule(rule, studiesAtCross, makeOhlcv([1, 2]))).toBe(true);

    // Un paso más (75 -> 80): sigue arriba del umbral, pero YA NO es el
    // cruce — no debe volver a disparar.
    const studiesAfterCross = makeStudies({ rsi_14: rsiSeries, fechas: ["2021-01-01", "2021-01-02", "2021-01-03"] });
    expect(evaluateRule(rule, studiesAfterCross, makeOhlcv([1, 2, 3]))).toBe(false);
  });

  it("rsi_below dispara solo al cruzar hacia abajo", () => {
    const rule = makeRule({ type: "rsi_below", threshold: 30 });

    expect(evaluateRule(rule, makeStudies({ rsi_14: [35, 25] }), makeOhlcv([1, 2]))).toBe(true);
    // Se mantiene por debajo: no es un cruce nuevo.
    expect(evaluateRule(rule, makeStudies({ rsi_14: [25, 20] }), makeOhlcv([1, 2]))).toBe(false);
    // Sube de nuevo: tampoco es un cruce hacia abajo.
    expect(evaluateRule(rule, makeStudies({ rsi_14: [20, 25] }), makeOhlcv([1, 2]))).toBe(false);
  });

  it("con menos de dos valores o el umbral en null, no evalúa (false)", () => {
    const rule = makeRule({ type: "rsi_above", threshold: 70 });
    expect(evaluateRule(rule, makeStudies({ rsi_14: [75], fechas: ["2021-01-01"] }), makeOhlcv([1]))).toBe(false);
    expect(evaluateRule(makeRule({ type: "rsi_above", threshold: null }), makeStudies({ rsi_14: [65, 75] }), makeOhlcv([1, 2]))).toBe(
      false,
    );
  });
});

describe("evaluateRule: stoch_above / stoch_below", () => {
  it("usa stoch_k (no stoch_d) y respeta la dirección del cruce", () => {
    const above = makeRule({ type: "stoch_above", threshold: 80 });
    expect(evaluateRule(above, makeStudies({ stoch_k: [70, 85] }), makeOhlcv([1, 2]))).toBe(true);
    expect(evaluateRule(above, makeStudies({ stoch_k: [85, 90] }), makeOhlcv([1, 2]))).toBe(false);

    const below = makeRule({ type: "stoch_below", threshold: 20 });
    expect(evaluateRule(below, makeStudies({ stoch_k: [25, 15] }), makeOhlcv([1, 2]))).toBe(true);
    expect(evaluateRule(below, makeStudies({ stoch_k: [15, 25] }), makeOhlcv([1, 2]))).toBe(false); // cruza para el otro lado
  });
});

describe("evaluateRule: price_cross_level_up / price_cross_level_down", () => {
  it("dispara cuando el CIERRE cruza el nivel elegido por el usuario", () => {
    const up = makeRule({ type: "price_cross_level_up", threshold: 100 });
    expect(evaluateRule(up, makeStudies(), makeOhlcv([95, 105]))).toBe(true);
    expect(evaluateRule(up, makeStudies(), makeOhlcv([105, 110]))).toBe(false); // ya venía arriba

    const down = makeRule({ type: "price_cross_level_down", threshold: 100 });
    expect(evaluateRule(down, makeStudies(), makeOhlcv([105, 95]))).toBe(true);
    expect(evaluateRule(down, makeStudies(), makeOhlcv([95, 90]))).toBe(false); // ya venía abajo
  });
});

describe("evaluateRule: price_cross_ma_up / price_cross_ma_down", () => {
  it("cruza la media elegida (maKey) hacia arriba/abajo", () => {
    const up = makeRule({ type: "price_cross_ma_up", maKey: "sma_50" });
    expect(evaluateRule(up, makeStudies({ sma_50: [100, 100] }), makeOhlcv([95, 105]))).toBe(true);
    expect(evaluateRule(up, makeStudies({ sma_20: [100, 100] }), makeOhlcv([95, 105]))).toBe(false); // maKey distinto no cuenta

    const down = makeRule({ type: "price_cross_ma_down", maKey: "ema_12" });
    expect(evaluateRule(down, makeStudies({ ema_12: [100, 100] }), makeOhlcv([105, 95]))).toBe(true);
  });

  it("sin maKey, no evalúa (false) en vez de romper", () => {
    const rule = makeRule({ type: "price_cross_ma_up", maKey: null });
    expect(evaluateRule(rule, makeStudies({ sma_20: [100, 100] }), makeOhlcv([95, 105]))).toBe(false);
  });
});

describe("evaluateRule: price_touch_sr / price_touch_poc / price_touch_value_area", () => {
  it("price_touch_sr dispara si la vela toca un soporte/resistencia con tolerancia 0.3%", () => {
    const rule = makeRule({ type: "price_touch_sr" });
    const studies = makeStudies({ soporte_resistencia: { resistencia: [110], soporte: [90], precio_actual: 100 } });
    // high=110.1 está a 0.09% de 110 -> dentro de tolerancia.
    expect(evaluateRule(rule, studies, makeOhlcv([100], [110.1], [99]))).toBe(true);
    // high=120 está lejos de cualquier nivel.
    expect(evaluateRule(rule, studies, makeOhlcv([100], [120], [99]))).toBe(false);
  });

  it("price_touch_poc necesita volumeProfile — sin él, no evalúa", () => {
    const rule = makeRule({ type: "price_touch_poc" });
    const ohlcv = makeOhlcv([100], [100.2], [99.8]);
    expect(evaluateRule(rule, makeStudies(), ohlcv)).toBe(false); // sin volumeProfile

    const profile: VolumeProfileResponse = {
      asset: "BTC",
      interval: "1d",
      niveles_precio: [100],
      volumenes: [1000],
      poc: 100,
      value_area_low: 95,
      value_area_high: 105,
      volumen_total: 1000,
    };
    expect(evaluateRule(rule, makeStudies(), ohlcv, profile)).toBe(true);
  });

  it("price_touch_value_area dispara al tocar CUALQUIERA de los dos bordes", () => {
    const rule = makeRule({ type: "price_touch_value_area" });
    const profile: VolumeProfileResponse = {
      asset: "BTC",
      interval: "1d",
      niveles_precio: [100],
      volumenes: [1000],
      poc: 100,
      value_area_low: 90,
      value_area_high: 110,
      volumen_total: 1000,
    };
    expect(evaluateRule(rule, makeStudies(), makeOhlcv([100], [110.05], [99]), profile)).toBe(true); // borde alto
    expect(evaluateRule(rule, makeStudies(), makeOhlcv([100], [101], [89.95]), profile)).toBe(true); // borde bajo
    expect(evaluateRule(rule, makeStudies(), makeOhlcv([100], [101], [99]), profile)).toBe(false); // en el medio
  });
});

describe("evaluateRule: macd_cross_signal_up / macd_cross_signal_down", () => {
  it("dispara cuando la línea MACD cruza su señal", () => {
    const up = makeRule({ type: "macd_cross_signal_up" });
    expect(
      evaluateRule(up, makeStudies({ macd: [-1, 1], macd_signal: [0, 0] }), makeOhlcv([1, 2])),
    ).toBe(true);
    expect(
      evaluateRule(up, makeStudies({ macd: [1, 2], macd_signal: [0, 0] }), makeOhlcv([1, 2])),
    ).toBe(false); // ya venía arriba de la señal

    const down = makeRule({ type: "macd_cross_signal_down" });
    expect(
      evaluateRule(down, makeStudies({ macd: [1, -1], macd_signal: [0, 0] }), makeOhlcv([1, 2])),
    ).toBe(true);
  });
});

describe("evaluateRule: bollinger_break_upper / bollinger_break_lower", () => {
  it("dispara cuando el cierre SALE de la banda (cruce, no solo estar afuera)", () => {
    const upper = makeRule({ type: "bollinger_break_upper" });
    expect(evaluateRule(upper, makeStudies({ bb_upper: [110, 110] }), makeOhlcv([105, 115]))).toBe(true);
    expect(evaluateRule(upper, makeStudies({ bb_upper: [110, 110] }), makeOhlcv([115, 120]))).toBe(false); // ya estaba afuera

    const lower = makeRule({ type: "bollinger_break_lower" });
    expect(evaluateRule(lower, makeStudies({ bb_lower: [90, 90] }), makeOhlcv([95, 85]))).toBe(true);
  });
});

describe("describeRule", () => {
  it("incluye el activo/intervalo de la regla (para listar reglas de cualquier moneda)", () => {
    const rule = makeRule({ asset: "ETH", interval: "1h", type: "rsi_above", threshold: 70 });
    expect(describeRule(rule)).toBe("ETH (1h) — RSI cruza por encima de 70");
  });

  it("arma el texto con el nombre de la media para price_cross_ma_*", () => {
    const rule = makeRule({ asset: "BTC", interval: "1d", type: "price_cross_ma_down", maKey: "ema_26" });
    expect(describeRule(rule)).toBe("BTC (1d) — Precio cruza EMA 26 hacia abajo");
  });
});
