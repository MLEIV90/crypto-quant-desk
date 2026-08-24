/**
 * Tarjeta de métrica reutilizada en las vistas de Riesgo/Research (Fase
 * 8c) — jerarquía visual consistente: etiqueta chica y muted arriba, valor
 * grande abajo, color opcional (p. ej. verde/rojo según dirección/régimen).
 *
 * `help` (Fase 9b, opcional): texto de `../helpTexts.ts` mostrado en un
 * `InfoTooltip` junto a la etiqueta — sin `help`, la tarjeta se ve
 * exactamente igual que antes (no todas las tarjetas lo necesitan).
 *
 * `subtext`/`subtextColor` (Fase 20a, opcional): una tercera línea chica
 * bajo el valor — pensada para el percentil histórico de una métrica de
 * riesgo ("percentil 70 — más alto que el 70% de la historia"), pero
 * genérica para cualquier otro dato secundario que necesite su propio
 * color (p. ej. ámbar/rojo cuando el percentil es alto).
 */

import type { ReactNode } from "react";
import { InfoTooltip } from "./InfoTooltip";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  valueColor?: string;
  help?: string;
  subtext?: ReactNode;
  subtextColor?: string;
}

export function MetricCard({ label, value, valueColor, help, subtext, subtextColor }: MetricCardProps) {
  return (
    <div className="metric-card">
      <span className="metric-card__label">
        {label}
        {help && <InfoTooltip text={help} placement="bottom" />}
      </span>
      <span className="metric-card__value" style={valueColor ? { color: valueColor } : undefined}>
        {value}
      </span>
      {subtext && (
        <span className="metric-card__subtext" style={subtextColor ? { color: subtextColor } : undefined}>
          {subtext}
        </span>
      )}
    </div>
  );
}
