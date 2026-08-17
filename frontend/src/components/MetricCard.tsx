/**
 * Tarjeta de métrica reutilizada en las vistas de Riesgo/Research (Fase
 * 8c) — jerarquía visual consistente: etiqueta chica y muted arriba, valor
 * grande abajo, color opcional (p. ej. verde/rojo según dirección/régimen).
 *
 * `help` (Fase 9b, opcional): texto de `../helpTexts.ts` mostrado en un
 * `InfoTooltip` junto a la etiqueta — sin `help`, la tarjeta se ve
 * exactamente igual que antes (no todas las tarjetas lo necesitan).
 */

import type { ReactNode } from "react";
import { InfoTooltip } from "./InfoTooltip";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  valueColor?: string;
  help?: string;
}

export function MetricCard({ label, value, valueColor, help }: MetricCardProps) {
  return (
    <div className="metric-card">
      <span className="metric-card__label">
        {label}
        {help && <InfoTooltip text={help} placement="bottom" />}
      </span>
      <span className="metric-card__value" style={valueColor ? { color: valueColor } : undefined}>
        {value}
      </span>
    </div>
  );
}
