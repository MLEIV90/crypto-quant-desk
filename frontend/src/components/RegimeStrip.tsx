/**
 * Franja de régimen de volatilidad a lo largo del tiempo (Fase 20a) —
 * `models.garch.volatility_regime` clasifica CADA fecha en calma/normal/
 * tensión (`GarchSeriesResponse.regimen_serie`); esto la pinta como una
 * franja horizontal continua (verde/gris/rojo, ver `REGIME_STRIP_COLORS`)
 * para ver de un vistazo cómo alternó el régimen en la historia y en qué
 * está ahora, sin tener que leer una tabla.
 *
 * CSS puro (segmentos posicionados proporcionalmente por índice de fecha),
 * no `lightweight-charts`: no hace falta un eje interactivo para esto,
 * solo una franja alineada con el mismo rango de fechas que el gráfico de
 * volatilidad condicional de arriba. Agrupa fechas consecutivas del MISMO
 * régimen en un solo segmento (en vez de un div por fecha) para no generar
 * miles de nodos DOM sobre varios años de historia diaria.
 */

import { REGIME_STRIP_COLORS } from "../theme";

type Regimen = "calma" | "normal" | "tension" | null;

interface RegimeStripProps {
  fechas: string[];
  regimenes: Regimen[];
  height?: number;
}

interface Segment {
  color: string;
  leftPct: number;
  widthPct: number;
}

function buildSegments(regimenes: Regimen[]): Segment[] {
  const n = regimenes.length;
  if (n === 0) return [];

  const segments: Segment[] = [];
  let start = 0;
  for (let i = 1; i <= n; i++) {
    if (i === n || regimenes[i] !== regimenes[start]) {
      const label = regimenes[start];
      segments.push({
        color: label ? REGIME_STRIP_COLORS[label] : "transparent",
        leftPct: (start / n) * 100,
        widthPct: ((i - start) / n) * 100,
      });
      start = i;
    }
  }
  return segments;
}

function formatFecha(iso: string): string {
  return new Date(iso).toLocaleDateString();
}

export function RegimeStrip({ fechas, regimenes, height = 24 }: RegimeStripProps) {
  if (fechas.length === 0) {
    return null;
  }

  const segments = buildSegments(regimenes);

  return (
    <div className="regime-strip">
      <div className="regime-strip__track" style={{ height }}>
        {segments.map((segment, index) => (
          <div
            key={index}
            className="regime-strip__segment"
            style={{ left: `${segment.leftPct}%`, width: `${segment.widthPct}%`, background: segment.color }}
          />
        ))}
      </div>
      <div className="regime-strip__axis">
        <span>{formatFecha(fechas[0])}</span>
        <span>{formatFecha(fechas[fechas.length - 1])}</span>
      </div>
      <div className="regime-strip__legend">
        <span className="regime-strip__legend-item">
          <i style={{ background: REGIME_STRIP_COLORS.calma }} /> Calma
        </span>
        <span className="regime-strip__legend-item">
          <i style={{ background: REGIME_STRIP_COLORS.normal }} /> Normal
        </span>
        <span className="regime-strip__legend-item">
          <i style={{ background: REGIME_STRIP_COLORS.tension }} /> Tensión
        </span>
      </div>
    </div>
  );
}
