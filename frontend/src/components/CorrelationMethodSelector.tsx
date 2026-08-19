/**
 * Selector de método de correlación (Fase 13b): Pearson (lineal, default) o
 * Spearman (de rangos). Reutiliza las clases CSS de `PeriodSelector`/
 * `ChartTypeSelector` (mismos pills) en vez de duplicar estilo.
 */

import { InfoTooltip } from "./InfoTooltip";
import { CORRELATION_METHOD_HELP } from "../helpTexts";

export type CorrelationMethodKey = "pearson" | "spearman";

export const DEFAULT_CORRELATION_METHOD: CorrelationMethodKey = "pearson";

const METHOD_OPTIONS: { key: CorrelationMethodKey; label: string }[] = [
  { key: "pearson", label: "Pearson" },
  { key: "spearman", label: "Spearman" },
];

interface CorrelationMethodSelectorProps {
  active: CorrelationMethodKey;
  onChange: (method: CorrelationMethodKey) => void;
}

export function CorrelationMethodSelector({ active, onChange }: CorrelationMethodSelectorProps) {
  return (
    <div className="period-selector" role="group" aria-label="Método de correlación">
      {METHOD_OPTIONS.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className={`period-selector__pill${active === key ? " period-selector__pill--active" : ""}`}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
      <InfoTooltip text={CORRELATION_METHOD_HELP} placement="bottom" />
    </div>
  );
}
