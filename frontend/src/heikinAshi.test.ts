/**
 * Tests de `toHeikinAshi` (Fase 14, hallazgo F-02 de auditoría) — fórmula
 * verificada contra un caso calculado A MANO, no contra el propio código
 * (si no, un bug en la fórmula pasaría desapercibido).
 */

import { describe, expect, it } from "vitest";
import { toHeikinAshi } from "./heikinAshi";
import type { Candle } from "./types";

function candle(fecha: string, open: number, high: number, low: number, close: number): Candle {
  return { fecha, open, high, low, close, volume: null };
}

describe("toHeikinAshi", () => {
  it("calcula la primera vela con HA_open = (open0 + close0) / 2 (no hay previa)", () => {
    const [ha] = toHeikinAshi([candle("2021-01-01", 100, 110, 95, 105)]);

    // HA_close = (100 + 110 + 95 + 105) / 4 = 102.5
    expect(ha.close).toBeCloseTo(102.5, 10);
    // HA_open, primera vela = (open + close) / 2 = (100 + 105) / 2 = 102.5
    expect(ha.open).toBeCloseTo(102.5, 10);
    // HA_high = max(high, HA_open, HA_close) = max(110, 102.5, 102.5)
    expect(ha.high).toBeCloseTo(110, 10);
    // HA_low = min(low, HA_open, HA_close) = min(95, 102.5, 102.5)
    expect(ha.low).toBeCloseTo(95, 10);
  });

  it("encadena HA_open_t = (HA_open_{t-1} + HA_close_{t-1}) / 2 en las velas siguientes", () => {
    const candles = [
      candle("2021-01-01", 100, 110, 95, 105),
      candle("2021-01-02", 105, 115, 100, 108),
      candle("2021-01-03", 108, 112, 104, 106),
    ];

    const result = toHeikinAshi(candles);

    // Vela 1 (ver el test anterior): HA_open=102.5, HA_close=102.5.
    // Vela 2: HA_close = (105+115+100+108)/4 = 107.
    //         HA_open  = (HA_open_1 + HA_close_1)/2 = (102.5+102.5)/2 = 102.5.
    //         HA_high  = max(115, 102.5, 107) = 115. HA_low = min(100, 102.5, 107) = 100.
    expect(result[1].close).toBeCloseTo(107, 10);
    expect(result[1].open).toBeCloseTo(102.5, 10);
    expect(result[1].high).toBeCloseTo(115, 10);
    expect(result[1].low).toBeCloseTo(100, 10);

    // Vela 3: HA_close = (108+112+104+106)/4 = 107.5.
    //         HA_open  = (HA_open_2 + HA_close_2)/2 = (102.5+107)/2 = 104.75.
    //         HA_high  = max(112, 104.75, 107.5) = 112. HA_low = min(104, 104.75, 107.5) = 104.
    expect(result[2].close).toBeCloseTo(107.5, 10);
    expect(result[2].open).toBeCloseTo(104.75, 10);
    expect(result[2].high).toBeCloseTo(112, 10);
    expect(result[2].low).toBeCloseTo(104, 10);
  });

  it("preserva la fecha y el volumen originales de cada vela (solo transforma OHLC)", () => {
    const [ha] = toHeikinAshi([{ fecha: "2021-01-01", open: 1, high: 2, low: 0.5, close: 1.5, volume: 999 }]);

    expect(ha.fecha).toBe("2021-01-01");
    expect(ha.volume).toBe(999);
  });

  it("devuelve la misma cantidad de velas que recibió, incluso con una sola vela o vacío", () => {
    expect(toHeikinAshi([])).toHaveLength(0);
    expect(toHeikinAshi([candle("2021-01-01", 1, 2, 0.5, 1.5)])).toHaveLength(1);
  });

  it("nunca inventa un rango: HA_high siempre >= HA_open/HA_close, HA_low siempre <= HA_open/HA_close", () => {
    // Vela bajista fuerte con mecha larga, para ejercitar que high/low
    // siguen envolviendo el cuerpo Heikin-Ashi incluso cuando el precio
    // real se movió mucho respecto del OHLC crudo.
    const candles = [
      candle("2021-01-01", 100, 105, 95, 98),
      candle("2021-01-02", 98, 99, 60, 62),
    ];
    const result = toHeikinAshi(candles);

    for (const ha of result) {
      expect(ha.high).toBeGreaterThanOrEqual(Math.max(ha.open, ha.close));
      expect(ha.low).toBeLessThanOrEqual(Math.min(ha.open, ha.close));
    }
  });
});
