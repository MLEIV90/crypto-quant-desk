/**
 * Tipos de dibujo sobre el gráfico (Fase 8d) — sin React, separado de
 * `components/DrawingTools.tsx` a propósito.
 *
 * Se guardan TODOS los dibujos de TODOS los activos/timeframes bajo una
 * única clave de localStorage (`cqd:drawings`), como un diccionario
 * indexado por `drawingsKey(asset, interval)` — así el hook de
 * persistencia usa una clave fija y cada consumidor filtra en memoria
 * (mismo criterio que `alerts.ts`/`AlertsPanel`).
 */

export type Drawing =
  | { id: string; kind: "horizontal"; price: number }
  | {
      id: string;
      kind: "trendline";
      pointA: { time: number; price: number };
      pointB: { time: number; price: number };
    };

export type DrawingsStore = Record<string, Drawing[]>;

export function drawingsKey(asset: string, interval: string): string {
  return `${asset}::${interval}`;
}
