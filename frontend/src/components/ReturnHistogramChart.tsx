/**
 * Histograma de retornos diarios con las marcas de VaR 95% y ES 95% sobre
 * la cola izquierda (Fase 20a) — NO reutiliza `BarChart.tsx` a propósito:
 * ese componente es para barras DIVERGENTES (positivo/negativo desde una
 * línea de cero, cada barra escalada de forma independiente, sin un eje
 * numérico real) — acá hace falta un eje de RETORNO real, para ubicar las
 * líneas de VaR/ES en la posición exacta que les corresponde entre los
 * bins, y barras todas apuntando para arriba (frecuencia) — un layout
 * distinto que no encaja en ese componente sin forzarlo. Mismo criterio de
 * "CSS puro, sin librería de gráficos" que `BarChart`/`MonthlyHeatmap`.
 */

import { COLORS } from "../theme";

interface ReturnHistogramChartProps {
  binEdges: number[];
  counts: number[];
  var95Return: number;
  es95Return: number;
  height?: number;
}

function pct(value: number, min: number, max: number): number {
  if (max <= min) return 0;
  return ((value - min) / (max - min)) * 100;
}

export function ReturnHistogramChart({
  binEdges,
  counts,
  var95Return,
  es95Return,
  height = 220,
}: ReturnHistogramChartProps) {
  if (binEdges.length < 2 || counts.length === 0) {
    return <p className="view-note">Sin datos suficientes para el histograma.</p>;
  }

  const min = binEdges[0];
  const max = binEdges[binEdges.length - 1];
  const maxCount = Math.max(...counts, 1);
  const varPct = pct(var95Return, min, max);
  const esPct = pct(es95Return, min, max);

  return (
    <div className="return-histogram">
      <div className="return-histogram__bars" style={{ height }}>
        {counts.map((count, index) => {
          const left = pct(binEdges[index], min, max);
          const right = pct(binEdges[index + 1], min, max);
          const binCenter = (binEdges[index] + binEdges[index + 1]) / 2;
          const isBadTail = binCenter <= var95Return;
          return (
            <div
              key={index}
              className="return-histogram__bar"
              title={`${(binEdges[index] * 100).toFixed(1)}% a ${(binEdges[index + 1] * 100).toFixed(1)}%: ${count} día${count === 1 ? "" : "s"}`}
              style={{
                left: `${left}%`,
                width: `${Math.max(right - left, 0.25)}%`,
                height: `${(count / maxCount) * 100}%`,
                background: isBadTail ? COLORS.danger : COLORS.accent,
              }}
            />
          );
        })}
        <div className="return-histogram__zero-line" style={{ left: `${pct(0, min, max)}%` }} />
        <div
          className="return-histogram__marker return-histogram__marker--var"
          style={{ left: `${varPct}%` }}
        >
          <span className="return-histogram__marker-label">VaR 95%</span>
        </div>
        <div
          className="return-histogram__marker return-histogram__marker--es"
          style={{ left: `${esPct}%` }}
        >
          <span className="return-histogram__marker-label">ES 95%</span>
        </div>
      </div>
      <div className="return-histogram__axis">
        <span>{(min * 100).toFixed(1)}%</span>
        <span>0%</span>
        <span>{(max * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}
