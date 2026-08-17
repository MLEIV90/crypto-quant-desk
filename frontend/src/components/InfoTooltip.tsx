/**
 * Ícono "?" con explicación al pasar el mouse o al enfocar con teclado
 * (Fase 9b) — componente único reutilizado en toda la app (StudyToggles,
 * OscillatorPanel, MetricCard, SuggesterPanel, BacktestView) para no
 * repetir el markup/CSS del tooltip en cada lugar. Los TEXTOS viven
 * centralizados en `../helpTexts.ts`, este componente no sabe nada de
 * contenido — solo de cómo mostrarlo.
 *
 * CSS-only (sin JS de posicionamiento): el bubble es hermano del botón
 * disparador dentro de un wrapper `position: relative`, y aparece con
 * `:hover`/`:focus-visible` (ver `.info-tooltip*` en App.css). `placement`
 * controla si el bubble se abre hacia arriba o hacia abajo, para no
 * recortarse contra los bordes de la página en secciones pegadas al tope
 * (toggles) o al fondo (headers de tabla).
 */

import { useId } from "react";

interface InfoTooltipProps {
  text: string;
  placement?: "top" | "bottom";
}

export function InfoTooltip({ text, placement = "top" }: InfoTooltipProps) {
  const id = useId();

  return (
    <span className="info-tooltip">
      <button
        type="button"
        className="info-tooltip__trigger"
        aria-label="Más información"
        aria-describedby={id}
      >
        ?
      </button>
      <span id={id} role="tooltip" className={`info-tooltip__bubble info-tooltip__bubble--${placement}`}>
        {text}
      </span>
    </span>
  );
}
