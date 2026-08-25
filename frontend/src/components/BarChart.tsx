/**
 * Barras "divergentes" (positivo hacia arriba, negativo hacia abajo desde
 * una línea de cero) — Fase 11, vista "Estadística". Los datos acá son
 * CATEGÓRICOS (mes, día de semana, rezago de ACF, bucket de periodograma),
 * no series temporales, así que no encajan en `lightweight-charts` (que
 * exige un eje de tiempo real, ver `Chart.tsx`/`LineChartPanel.tsx`) — CSS
 * puro en vez de forzar una librería de gráficos financieros para esto.
 *
 * `horizontal` (Fase 24, vista "Research"): variante de filas horizontales
 * en vez de columnas verticales — pensada para listas cortas de valores NO
 * divergentes (todos positivos, p. ej. importancia de features o accuracy
 * vs. baselines), donde una barra horizontal con el valor al lado se lee
 * más directo que una columna que solo usa la mitad superior del alto
 * disponible. Mismo `BarChartDatum`, sin cambiar el modo vertical existente.
 */

import { COLORS } from "../theme";

export interface BarChartDatum {
  label: string;
  value: number;
  title?: string;
}

interface BarChartProps {
  data: BarChartDatum[];
  height?: number;
  positiveColor?: string;
  negativeColor?: string;
  horizontal?: boolean;
  /** Solo para `horizontal`: formatea el valor mostrado al final de cada barra. */
  formatValue?: (value: number) => string;
}

export function BarChart({
  data,
  height = 160,
  positiveColor = COLORS.success,
  negativeColor = COLORS.danger,
  horizontal = false,
  formatValue,
}: BarChartProps) {
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.value)), 1e-9);

  if (horizontal) {
    return (
      <div className="bar-chart bar-chart--horizontal">
        {data.map((d, index) => {
          const pct = (Math.abs(d.value) / maxAbs) * 100;
          const color = d.value >= 0 ? positiveColor : negativeColor;
          return (
            <div key={`${d.label}-${index}`} className="bar-chart__hrow" title={d.title}>
              <span className="bar-chart__hlabel">{d.label}</span>
              <div className="bar-chart__htrack">
                <div className="bar-chart__hbar" style={{ width: `${pct}%`, background: color }} />
              </div>
              <span className="bar-chart__hvalue">{formatValue ? formatValue(d.value) : d.value.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="bar-chart">
      <div className="bar-chart__plot" style={{ height }}>
        <div className="bar-chart__zero-line" />
        {data.map((d, index) => {
          const pct = (Math.abs(d.value) / maxAbs) * 50;
          const isPositive = d.value >= 0;
          return (
            <div key={`${d.label}-${index}`} className="bar-chart__col" title={d.title}>
              <div
                className="bar-chart__bar"
                style={
                  isPositive
                    ? { height: `${pct}%`, bottom: "50%", background: positiveColor }
                    : { height: `${pct}%`, top: "50%", background: negativeColor }
                }
              />
            </div>
          );
        })}
      </div>
      <div className="bar-chart__labels">
        {data.map((d, index) => (
          <span key={`${d.label}-${index}-label`} className="bar-chart__label">
            {d.label}
          </span>
        ))}
      </div>
    </div>
  );
}
