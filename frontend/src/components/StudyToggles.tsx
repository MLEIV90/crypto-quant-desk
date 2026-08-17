/**
 * Checkboxes para prender/apagar los OVERLAYS que se dibujan sobre el
 * precio (`components/Chart.tsx` los lee de `/api/studies`, Fase 7a/8a) —
 * solo UI de toggle, no toca lightweight-charts directamente.
 *
 * CLAVE (pedido explícito de la Fase 8b): por defecto se muestran POCOS
 * overlays (ver `DEFAULT_ACTIVE_OVERLAYS`, usado por `App.tsx`) — nunca
 * todo encimado de entrada como el gráfico matplotlib de la Fase 7b; el
 * usuario agrega lo que quiera desde acá.
 *
 * Cada chip lleva un `InfoTooltip` (Fase 9b) con el texto de
 * `../helpTexts.ts::OVERLAY_HELP` — qué mide cada overlay y cómo leerlo,
 * redactado para no sugerir que algo sin edge demostrado predice el precio.
 */

import { InfoTooltip } from "./InfoTooltip";
import { OVERLAY_HELP } from "../helpTexts";

export type OverlayKey =
  | "sma20"
  | "sma50"
  | "ema12"
  | "ema26"
  | "bollinger"
  | "fibonacci"
  | "supportResistance"
  | "pivots";

export const DEFAULT_ACTIVE_OVERLAYS: OverlayKey[] = ["sma20", "sma50"];

const OVERLAY_OPTIONS: { key: OverlayKey; label: string }[] = [
  { key: "sma20", label: "SMA 20" },
  { key: "sma50", label: "SMA 50" },
  { key: "ema12", label: "EMA 12" },
  { key: "ema26", label: "EMA 26" },
  { key: "bollinger", label: "Bollinger" },
  { key: "fibonacci", label: "Fibonacci" },
  { key: "supportResistance", label: "Soporte/Resistencia" },
  { key: "pivots", label: "Pivotes" },
];

interface StudyTogglesProps {
  active: Set<OverlayKey>;
  onToggle: (key: OverlayKey) => void;
}

export function StudyToggles({ active, onToggle }: StudyTogglesProps) {
  return (
    <fieldset className="toggles">
      <legend>Overlays sobre el precio</legend>
      {OVERLAY_OPTIONS.map(({ key, label }) => (
        <label key={key} className={`toggle-chip${active.has(key) ? " toggle-chip--active" : ""}`}>
          <input type="checkbox" checked={active.has(key)} onChange={() => onToggle(key)} />
          {label}
          <InfoTooltip text={OVERLAY_HELP[key]} placement="bottom" />
        </label>
      ))}
    </fieldset>
  );
}
