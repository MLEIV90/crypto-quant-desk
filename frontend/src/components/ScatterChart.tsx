/**
 * Scatter (dispersión) con recta de regresión (Fase 15b, vista "Arbitraje")
 * — `lightweight-charts` es una librería de series TEMPORALES, no tiene
 * scatter plot; en vez de sumar una librería nueva para un solo gráfico
 * simple, esto es SVG puro (sin dependencias) — un puñado de círculos +
 * una línea, no hace falta más.
 *
 * Grafica log(precio X) en el eje horizontal vs. log(precio Y) en el
 * vertical — NO precios crudos: `alpha`/`beta` (la recta) salen de
 * `pairs.cointegration.engle_granger`, una regresión sobre LOG-precios
 * (`log(y) = alpha + beta*log(x) + spread`, ver ese docstring). Graficar la
 * misma recta contra precios crudos la mostraría desalineada de los
 * puntos — no es la regresión que realmente se ajustó.
 */

import { useMemo } from "react";
import { COLORS } from "../theme";

export interface ScatterPoint {
  x: number;
  y: number;
}

interface ScatterChartProps {
  points: ScatterPoint[];
  alpha: number;
  beta: number;
  xLabel: string;
  yLabel: string;
  height?: number;
}

const PADDING = { top: 16, right: 16, bottom: 36, left: 52 };

export function ScatterChart({ points, alpha, beta, xLabel, yLabel, height = 320 }: ScatterChartProps) {
  const width = 640; // viewBox lógico — el SVG escala a 100% del contenedor real

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xLine = [Math.min(...xs), Math.max(...xs)].map((x) => alpha + beta * x);
    const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.05 || 1;
    const yValuesWithLine = [...ys, ...xLine];
    const yPad = (Math.max(...yValuesWithLine) - Math.min(...yValuesWithLine)) * 0.08 || 1;
    return {
      xMin: Math.min(...xs) - xPad,
      xMax: Math.max(...xs) + xPad,
      yMin: Math.min(...yValuesWithLine) - yPad,
      yMax: Math.max(...yValuesWithLine) + yPad,
    };
  }, [points, alpha, beta]);

  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;

  function toSvgX(x: number): number {
    return PADDING.left + ((x - xMin) / (xMax - xMin || 1)) * plotWidth;
  }
  function toSvgY(y: number): number {
    return PADDING.top + (1 - (y - yMin) / (yMax - yMin || 1)) * plotHeight;
  }

  const lineX1 = xMin;
  const lineX2 = xMax;
  const lineY1 = alpha + beta * lineX1;
  const lineY2 = alpha + beta * lineX2;

  const xTicks = [xMin, (xMin + xMax) / 2, xMax];
  const yTicks = [yMin, (yMin + yMax) / 2, yMax];

  if (points.length === 0) {
    return (
      <div className="chart-container" style={{ height, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span className="view-note">Sin datos suficientes para el scatter.</span>
      </div>
    );
  }

  return (
    <div className="chart-container" style={{ height, padding: 0 }}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" aria-label="Scatter con recta de regresión">
        <rect x={0} y={0} width={width} height={height} fill={COLORS.background} />

        {/* Grid + ticks */}
        {xTicks.map((t) => (
          <line key={`vx-${t}`} x1={toSvgX(t)} x2={toSvgX(t)} y1={PADDING.top} y2={height - PADDING.bottom} stroke={COLORS.border} strokeWidth={1} />
        ))}
        {yTicks.map((t) => (
          <line key={`vy-${t}`} x1={PADDING.left} x2={width - PADDING.right} y1={toSvgY(t)} y2={toSvgY(t)} stroke={COLORS.border} strokeWidth={1} />
        ))}
        {xTicks.map((t) => (
          <text key={`tx-${t}`} x={toSvgX(t)} y={height - PADDING.bottom + 16} fill={COLORS.textFaint} fontSize={10} textAnchor="middle">
            {t.toFixed(2)}
          </text>
        ))}
        {yTicks.map((t) => (
          <text key={`ty-${t}`} x={PADDING.left - 8} y={toSvgY(t) + 3} fill={COLORS.textFaint} fontSize={10} textAnchor="end">
            {t.toFixed(2)}
          </text>
        ))}

        {/* Puntos */}
        {points.map((p, i) => (
          <circle key={i} cx={toSvgX(p.x)} cy={toSvgY(p.y)} r={2.5} fill={COLORS.accent} opacity={0.45} />
        ))}

        {/* Recta de regresión */}
        <line
          x1={toSvgX(lineX1)}
          y1={toSvgY(lineY1)}
          x2={toSvgX(lineX2)}
          y2={toSvgY(lineY2)}
          stroke={COLORS.volumeProfilePoc}
          strokeWidth={2}
        />

        {/* Ejes */}
        <text x={width / 2} y={height - 4} fill={COLORS.textMuted} fontSize={11} textAnchor="middle">
          {xLabel}
        </text>
        <text
          x={12}
          y={height / 2}
          fill={COLORS.textMuted}
          fontSize={11}
          textAnchor="middle"
          transform={`rotate(-90, 12, ${height / 2})`}
        >
          {yLabel}
        </text>
      </svg>
    </div>
  );
}
