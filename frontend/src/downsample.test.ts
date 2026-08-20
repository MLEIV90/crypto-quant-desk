/**
 * Tests de `downsample.ts` (Fase 14, hallazgos F-02 + F-04 de auditoría):
 * la propiedad crítica es que el ÚLTIMO punto real (el precio/nivel más
 * reciente) nunca se pierda, y que por debajo del umbral no se toque nada.
 */

import { describe, expect, it } from "vitest";
import { computeDownsampleStep, DOWNSAMPLE_THRESHOLD, downsampleByStep } from "./downsample";

describe("computeDownsampleStep", () => {
  it("devuelve 1 (sin downsample) por debajo o igual al umbral", () => {
    expect(computeDownsampleStep(100, 10_000, 6_000)).toBe(1);
    expect(computeDownsampleStep(10_000, 10_000, 6_000)).toBe(1);
  });

  it("por encima del umbral, devuelve un step que deja aprox. 'target' puntos", () => {
    const step = computeDownsampleStep(58_000, 10_000, 6_000);
    expect(step).toBeGreaterThan(1);
    expect(Math.ceil(58_000 / step)).toBeLessThanOrEqual(6_000);
  });

  it("usa las constantes por defecto si no se pasan threshold/target", () => {
    expect(computeDownsampleStep(DOWNSAMPLE_THRESHOLD - 1)).toBe(1);
    expect(computeDownsampleStep(DOWNSAMPLE_THRESHOLD + 1)).toBeGreaterThan(1);
  });
});

describe("downsampleByStep", () => {
  it("con step <= 1, devuelve el array SIN copiar (misma referencia)", () => {
    const items = [1, 2, 3];
    expect(downsampleByStep(items, 1)).toBe(items);
    expect(downsampleByStep(items, 0)).toBe(items);
  });

  it("toma 1 de cada 'step' elementos, en orden", () => {
    const items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(downsampleByStep(items, 3)).toEqual([0, 3, 6, 9]);
  });

  it("SIEMPRE conserva el último elemento, aunque no caiga justo en el paso", () => {
    const items = [0, 1, 2, 3, 4, 5, 6, 7]; // step=3 -> índices 0,3,6, último real es índice 7
    const result = downsampleByStep(items, 3);
    expect(result[result.length - 1]).toBe(7);
    expect(result).toEqual([0, 3, 6, 7]);
  });

  it("no duplica el último elemento si ya cayó exacto en el paso", () => {
    const items = [0, 1, 2, 3, 4, 5, 6]; // step=3 -> índices 0,3,6, y 6 ya es el último índice
    expect(downsampleByStep(items, 3)).toEqual([0, 3, 6]);
  });

  it("con un array vacío, devuelve vacío sin romper", () => {
    expect(downsampleByStep([], 5)).toEqual([]);
  });

  it("reduce ~58.000 puntos a un rango manejable preservando el último", () => {
    const items = Array.from({ length: 58_000 }, (_, i) => i);
    const step = computeDownsampleStep(items.length);
    const result = downsampleByStep(items, step);

    expect(result.length).toBeLessThan(8_000);
    expect(result.length).toBeGreaterThan(1_000);
    expect(result[result.length - 1]).toBe(57_999);
  });
});
