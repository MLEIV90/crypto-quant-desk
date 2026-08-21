/**
 * Mapa de calor mes x año de retornos (Fase 15a) — mismo patrón de grilla
 * simple que `CorrelationHeatmap.tsx` (Fase 13b: divs con color de fondo
 * según el valor, sin librería nueva), pero con su propia escala de color:
 * la correlación está acotada a [-1, 1], el retorno mensual NO (puede ir de
 * -80% a +300% en cripto) — acá se usa un tope configurable (`capPct`) más
 * allá del cual el color ya no se satura más, para que un solo mes extremo
 * no aplaste visualmente la escala del resto.
 */

import { Fragment } from "react";

interface Rgb {
  r: number;
  g: number;
  b: number;
}

const WHITE: Rgb = { r: 255, g: 255, b: 255 };
const GREEN: Rgb = { r: 21, g: 128, b: 61 };
const RED: Rgb = { r: 185, g: 28, b: 28 };

const MONTH_LABELS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"];

const DEFAULT_CAP_PCT = 30;

function monthlyColor(pct: number, capPct: number): Rgb {
  const clamped = Math.max(-capPct, Math.min(capPct, pct)) / capPct;
  const target = clamped >= 0 ? GREEN : RED;
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

interface MonthlyHeatmapProps {
  anios: number[];
  matriz: (number | null)[][];
  capPct?: number;
}

export function MonthlyHeatmap({ anios, matriz, capPct = DEFAULT_CAP_PCT }: MonthlyHeatmapProps) {
  return (
    <div
      className="correlation-heatmap"
      style={{ gridTemplateColumns: `auto repeat(${anios.length}, 1fr)` }}
      role="table"
      aria-label="Mapa de calor de retornos mes por año"
    >
      <div className="correlation-heatmap__corner" />
      {anios.map((year) => (
        <div key={`col-${year}`} className="correlation-heatmap__col-label">
          {year}
        </div>
      ))}
      {MONTH_LABELS.map((monthLabel, monthIndex) => (
        <Fragment key={monthLabel}>
          <div className="correlation-heatmap__row-label">{monthLabel}</div>
          {anios.map((year, yearIndex) => {
            const value = matriz[monthIndex]?.[yearIndex] ?? null;
            if (value === null) {
              return (
                <div key={year} className="correlation-heatmap__cell correlation-heatmap__cell--empty">
                  —
                </div>
              );
            }
            const bg = monthlyColor(value, capPct);
            const sign = value >= 0 ? "+" : "";
            return (
              <div
                key={year}
                className="correlation-heatmap__cell"
                style={{ background: `rgb(${bg.r}, ${bg.g}, ${bg.b})`, color: textColorFor(bg) }}
                title={`${monthLabel} ${year}: ${sign}${value.toFixed(2)}%`}
              >
                {sign}
                {value.toFixed(0)}%
              </div>
            );
          })}
        </Fragment>
      ))}
    </div>
  );
}
