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
 *
 * Fase 10b: VWAP e Ichimoku se agregan en un grupo "Avanzados" aparte,
 * separado visualmente de los 8 overlays clásicos — con 10 opciones en
 * una sola fila plana el panel se saturaba.
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
  | "pivots"
  | "vwap"
  | "ichimoku";

export const DEFAULT_ACTIVE_OVERLAYS: OverlayKey[] = ["sma20", "sma50"];

const BASIC_OVERLAY_OPTIONS: { key: OverlayKey; label: string }[] = [
  { key: "sma20", label: "SMA 20" },
  { key: "sma50", label: "SMA 50" },
  { key: "ema12", label: "EMA 12" },
  { key: "ema26", label: "EMA 26" },
  { key: "bollinger", label: "Bollinger" },
  { key: "fibonacci", label: "Fibonacci" },
  { key: "supportResistance", label: "Soporte/Resistencia" },
  { key: "pivots", label: "Pivotes" },
];

const ADVANCED_OVERLAY_OPTIONS: { key: OverlayKey; label: string }[] = [
  { key: "vwap", label: "VWAP" },
  { key: "ichimoku", label: "Ichimoku" },
];

interface StudyTogglesProps {
  active: Set<OverlayKey>;
  onToggle: (key: OverlayKey) => void;
}

function OverlayChip({
  option,
  active,
  onToggle,
}: {
  option: { key: OverlayKey; label: string };
  active: Set<OverlayKey>;
  onToggle: (key: OverlayKey) => void;
}) {
  const { key, label } = option;
  return (
    <label className={`toggle-chip${active.has(key) ? " toggle-chip--active" : ""}`}>
      <input type="checkbox" checked={active.has(key)} onChange={() => onToggle(key)} />
      {label}
      <InfoTooltip text={OVERLAY_HELP[key]} placement="bottom" />
    </label>
  );
}

export function StudyToggles({ active, onToggle }: StudyTogglesProps) {
  return (
    <fieldset className="toggles">
      <legend>Overlays sobre el precio</legend>
      <div className="toggles__group">
        {BASIC_OVERLAY_OPTIONS.map((option) => (
          <OverlayChip key={option.key} option={option} active={active} onToggle={onToggle} />
        ))}
      </div>
      <div className="toggles__group toggles__group--advanced">
        <span className="toggles__group-label">Avanzados</span>
        {ADVANCED_OVERLAY_OPTIONS.map((option) => (
          <OverlayChip key={option.key} option={option} active={active} onToggle={onToggle} />
        ))}
      </div>
    </fieldset>
  );
}
