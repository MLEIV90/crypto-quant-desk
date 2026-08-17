/**
 * Checkboxes para prender/apagar los OSCILADORES (RSI, MACD, Estocástico),
 * cada uno dibujado por `components/Chart.tsx` en su propio PANE de
 * lightweight-charts, sincronizado en el eje temporal con el precio — acá
 * solo vive el toggle, no el gráfico en sí (los panes tienen que vivir en
 * la MISMA instancia de chart que las velas para estar sincronizados, así
 * que la creación/destrucción del pane la maneja `Chart.tsx`).
 *
 * Todos apagados por defecto: igual que los overlays (ver `StudyToggles`),
 * el usuario agrega los osciladores que quiera en vez de arrancar con todo
 * encimado.
 *
 * Cada chip lleva un `InfoTooltip` (Fase 9b) con el texto de
 * `../helpTexts.ts::OSCILLATOR_HELP`.
 */

import { InfoTooltip } from "./InfoTooltip";
import { OSCILLATOR_HELP } from "../helpTexts";

export type OscillatorKey = "rsi" | "macd" | "stochastic";

export const DEFAULT_ACTIVE_OSCILLATORS: OscillatorKey[] = [];

const OSCILLATOR_OPTIONS: { key: OscillatorKey; label: string }[] = [
  { key: "rsi", label: "RSI (30/70)" },
  { key: "macd", label: "MACD" },
  { key: "stochastic", label: "Estocástico (20/80)" },
];

interface OscillatorPanelProps {
  active: Set<OscillatorKey>;
  onToggle: (key: OscillatorKey) => void;
}

export function OscillatorPanel({ active, onToggle }: OscillatorPanelProps) {
  return (
    <fieldset className="toggles">
      <legend>Osciladores (paneles inferiores)</legend>
      {OSCILLATOR_OPTIONS.map(({ key, label }) => (
        <label key={key} className={`toggle-chip${active.has(key) ? " toggle-chip--active" : ""}`}>
          <input type="checkbox" checked={active.has(key)} onChange={() => onToggle(key)} />
          {label}
          <InfoTooltip text={OSCILLATOR_HELP[key]} placement="bottom" />
        </label>
      ))}
    </fieldset>
  );
}
