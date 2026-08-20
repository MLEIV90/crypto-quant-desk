/**
 * Downsampling VISUAL de series largas (Fase 14, hallazgo F-04 de
 * auditoría): con intervalo horario y el período "Todo" (~58.000 velas,
 * ver `components/PeriodSelector.tsx`), dibujar cada punto individual
 * puede ir lento en hardware modesto. Por encima de `DOWNSAMPLE_THRESHOLD`
 * puntos, `components/Chart.tsx` toma 1 de cada `step` SOLO para el dibujo
 * (siempre conservando el último punto, para que el precio/nivel más
 * reciente que se ve en el gráfico sea el real) — es una decisión de
 * RENDERIZADO nada más.
 *
 * Los cálculos (`/api/studies`, `/api/risk`, `AlertsPanel`, etc.) NUNCA
 * pasan por acá — siguen leyendo `ohlcv`/`studies` tal cual llegan de la
 * API, con todos los datos reales. Este módulo solo lo usa `Chart.tsx`,
 * justo antes de `series.setData()`.
 */

export const DOWNSAMPLE_THRESHOLD = 10_000;
export const DOWNSAMPLE_TARGET_POINTS = 6_000;

/** `step` a usar para que `totalPoints` queden en, aprox, `target` puntos
 * dibujados — 1 (sin downsample) si `totalPoints` no supera `threshold`.
 */
export function computeDownsampleStep(
  totalPoints: number,
  threshold: number = DOWNSAMPLE_THRESHOLD,
  target: number = DOWNSAMPLE_TARGET_POINTS,
): number {
  if (totalPoints <= threshold) return 1;
  return Math.ceil(totalPoints / target);
}

/** Toma 1 de cada `step` elementos de `items`, preservando el orden y
 * SIEMPRE incluyendo el último elemento (aunque no caiga en el paso) —
 * para que la vista nunca "recorte" el dato más reciente. `step <= 1`
 * devuelve `items` sin copiar.
 */
export function downsampleByStep<T>(items: T[], step: number): T[] {
  if (step <= 1 || items.length === 0) return items;

  const result: T[] = [];
  for (let i = 0; i < items.length; i += step) {
    result.push(items[i]);
  }
  const lastIndex = items.length - 1;
  if (result[result.length - 1] !== items[lastIndex]) {
    result.push(items[lastIndex]);
  }
  return result;
}
