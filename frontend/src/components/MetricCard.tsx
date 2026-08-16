/**
 * Tarjeta de métrica reutilizada en las vistas de Riesgo/Research (Fase
 * 8c) — jerarquía visual consistente: etiqueta chica y muted arriba, valor
 * grande abajo, color opcional (p. ej. verde/rojo según dirección/régimen).
 */

import type { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  valueColor?: string;
}

export function MetricCard({ label, value, valueColor }: MetricCardProps) {
  return (
    <div className="metric-card">
      <span className="metric-card__label">{label}</span>
      <span className="metric-card__value" style={valueColor ? { color: valueColor } : undefined}>
        {value}
      </span>
    </div>
  );
}
