/**
 * Mapa de calor 5x5 de correlación (Fase 13b) — grilla simple de divs con
 * color de fondo según el valor (sin librería nueva, pedido explícito de
 * la fase): gradiente blanco (0) -> azul (+1) / blanco (0) -> rojo (-1).
 * El color de texto de cada celda se elige por luminancia del fondo (fondos
 * casi blancos = texto oscuro, fondos saturados = texto claro) para que el
 * número quede legible en toda la escala.
 */

import { Fragment } from "react";

interface Rgb {
  r: number;
  g: number;
  b: number;
}

const WHITE: Rgb = { r: 255, g: 255, b: 255 };
const BLUE: Rgb = { r: 29, g: 78, b: 216 };
const RED: Rgb = { r: 220, g: 38, b: 38 };

function correlationColor(value: number): Rgb {
  const clamped = Math.max(-1, Math.min(1, value));
  const target = clamped >= 0 ? BLUE : RED;
  const t = Math.abs(clamped);
  return {
    r: Math.round(WHITE.r + (target.r - WHITE.r) * t),
    g: Math.round(WHITE.g + (target.g - WHITE.g) * t),
    b: Math.round(WHITE.b + (target.b - WHITE.b) * t),
  };
}

function textColorFor({ r, g, b }: Rgb): string {
  const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
  return luminance > 140 ? "#111827" : "#f8fafc";
}

interface CorrelationHeatmapProps {
  activos: string[];
  matriz: (number | null)[][];
}

export function CorrelationHeatmap({ activos, matriz }: CorrelationHeatmapProps) {
  return (
    <div
      className="correlation-heatmap"
      style={{ gridTemplateColumns: `auto repeat(${activos.length}, 1fr)` }}
      role="table"
      aria-label="Mapa de calor de correlación entre activos"
    >
      <div className="correlation-heatmap__corner" />
      {activos.map((asset) => (
        <div key={`col-${asset}`} className="correlation-heatmap__col-label">
          {asset}
        </div>
      ))}
      {activos.map((rowAsset, i) => (
        <Fragment key={rowAsset}>
          <div className="correlation-heatmap__row-label">{rowAsset}</div>
          {activos.map((colAsset, j) => {
            const value = matriz[i]?.[j] ?? null;
            if (value === null) {
              return (
                <div key={colAsset} className="correlation-heatmap__cell correlation-heatmap__cell--empty">
                  —
                </div>
              );
            }
            const bg = correlationColor(value);
            return (
              <div
                key={colAsset}
                className="correlation-heatmap__cell"
                style={{ background: `rgb(${bg.r}, ${bg.g}, ${bg.b})`, color: textColorFor(bg) }}
                title={`${rowAsset} vs ${colAsset}: ${value.toFixed(3)}`}
              >
                {value.toFixed(2)}
              </div>
            );
          })}
        </Fragment>
      ))}
    </div>
  );
}
