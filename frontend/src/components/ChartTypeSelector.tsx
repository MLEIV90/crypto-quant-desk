/**
 * Selector de tipo de gráfico (Fase 10b): velas japonesas (el default de
 * siempre) o Heikin-Ashi (`../heikinAshi.ts`, calculado en el frontend).
 * Reutiliza las clases CSS de `PeriodSelector` (mismos pills, ver App.css)
 * en vez de duplicar el estilo — visualmente son el mismo tipo de control.
 */

import { InfoTooltip } from "./InfoTooltip";
import { CHART_TYPE_HELP } from "../helpTexts";

export type ChartTypeKey = "candles" | "heikin-ashi";

export const DEFAULT_CHART_TYPE: ChartTypeKey = "candles";

const CHART_TYPE_OPTIONS: { key: ChartTypeKey; label: string }[] = [
  { key: "candles", label: "Velas" },
  { key: "heikin-ashi", label: "Heikin-Ashi" },
];

interface ChartTypeSelectorProps {
  active: ChartTypeKey;
  onChange: (type: ChartTypeKey) => void;
}

export function ChartTypeSelector({ active, onChange }: ChartTypeSelectorProps) {
  return (
    <div className="period-selector" role="group" aria-label="Tipo de gráfico">
      {CHART_TYPE_OPTIONS.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className={`period-selector__pill${active === key ? " period-selector__pill--active" : ""}`}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
      <InfoTooltip text={CHART_TYPE_HELP} placement="bottom" />
    </div>
  );
}
