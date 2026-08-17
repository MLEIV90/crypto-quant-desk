/**
 * Herramientas de dibujo sobre el gráfico de velas (Fase 8d).
 *
 * ENFOQUE (documentado según lo pedido): `lightweight-charts` en su
 * versión gratuita (la que usa este proyecto, ver `package.json`) no trae
 * "line tools" interactivas — eso es parte de un plugin comercial
 * separado, no de la librería open-source. Para líneas horizontales se
 * reutiliza la primitiva nativa `series.createPriceLine()` (la misma que
 * `Chart.tsx` ya usa para Fibonacci/soporte-resistencia/pivotes). Para
 * líneas de tendencia NO hay primitiva nativa: se arman a mano con una
 * `LineSeries` normal de 2 puntos, tomando el click del usuario sobre el
 * gráfico vía `chart.subscribeClick()` y traduciendo el pixel a precio
 * real con `series.coordinateToPrice()` — 100% con la API pública
 * gratuita, sin dependencias nuevas.
 *
 * Los dibujos persisten en `localStorage`, separados por activo/timeframe
 * (ver `../drawings.ts`), y sobreviven a cambiar de vista/pestaña.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { LineSeries, LineStyle } from "lightweight-charts";
import type { IChartApi, ISeriesApi, MouseEventParams, Time, UTCTimestamp } from "lightweight-charts";
import { useLocalStorageState } from "../hooks/useLocalStorageState";
import { drawingsKey, type Drawing, type DrawingsStore } from "../drawings";
import { COLORS } from "../theme";

const STORAGE_KEY = "cqd:drawings";

type DrawMode = "idle" | "horizontal" | "trendline-1" | "trendline-2";

interface DrawingToolsProps {
  asset: string;
  interval: string;
  chart: IChartApi | null;
  candleSeries: ISeriesApi<"Candlestick", Time> | null;
}

interface RenderedArtifact {
  remove: () => void;
}

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function DrawingTools({ asset, interval, chart, candleSeries }: DrawingToolsProps) {
  const [store, setStore] = useLocalStorageState<DrawingsStore>(STORAGE_KEY, {});
  const key = drawingsKey(asset, interval);
  const drawings = useMemo(() => store[key] ?? [], [store, key]);

  const [mode, setModeState] = useState<DrawMode>("idle");
  const modeRef = useRef<DrawMode>("idle");
  const pendingPointRef = useRef<{ time: UTCTimestamp; price: number } | null>(null);
  const artifactsRef = useRef<RenderedArtifact[]>([]);

  function setMode(next: DrawMode) {
    modeRef.current = next;
    setModeState(next);
  }

  function persist(next: Drawing[]) {
    setStore((prev) => ({ ...prev, [key]: next }));
  }

  // Cambiar de activo/timeframe a mitad de un dibujo dejaría un punto
  // pendiente con coordenadas de OTRA escala de precio — se cancela.
  useEffect(() => {
    setMode("idle");
    pendingPointRef.current = null;
  }, [asset, interval]);

  // Suscripción de click: se registra una vez por instancia de chart/serie
  // disponibles. Usa `modeRef` (no el estado de React) para decidir qué
  // hacer, así nunca queda pegada a un modo viejo por un closure stale.
  useEffect(() => {
    if (!chart || !candleSeries) return;
    const series = candleSeries;

    function handleClick(param: MouseEventParams<Time>) {
      if (modeRef.current === "idle") return;
      if (!param.point || param.time === undefined) return;
      const price = series.coordinateToPrice(param.point.y);
      if (price === null) return;
      const time = param.time as UTCTimestamp;

      if (modeRef.current === "horizontal") {
        persist([...drawings, { id: createId(), kind: "horizontal", price }]);
        setMode("idle");
        return;
      }
      if (modeRef.current === "trendline-1") {
        pendingPointRef.current = { time, price };
        setMode("trendline-2");
        return;
      }
      if (modeRef.current === "trendline-2") {
        const pointA = pendingPointRef.current;
        pendingPointRef.current = null;
        setMode("idle");
        if (!pointA) return;
        persist([...drawings, { id: createId(), kind: "trendline", pointA, pointB: { time, price } }]);
      }
    }

    chart.subscribeClick(handleClick);
    return () => chart.unsubscribeClick(handleClick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chart, candleSeries, drawings]);

  // Render: reconstruye TODAS las primitivas dibujadas cuando cambian los
  // datos guardados o el chart disponible — mismo criterio (reconstrucción
  // completa) que usa Chart.tsx para overlays/osciladores.
  useEffect(() => {
    if (!chart || !candleSeries) return;

    artifactsRef.current.forEach((artifact) => artifact.remove());
    artifactsRef.current = [];

    drawings.forEach((drawing) => {
      if (drawing.kind === "horizontal") {
        const line = candleSeries.createPriceLine({
          price: drawing.price,
          color: COLORS.accent,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title: "dibujo",
        });
        artifactsRef.current.push({ remove: () => candleSeries.removePriceLine(line) });
      } else {
        const lineSeries = chart.addSeries(LineSeries, {
          color: COLORS.accent,
          lineWidth: 2,
          lastValueVisible: false,
          priceLineVisible: false,
        });
        lineSeries.setData([
          { time: drawing.pointA.time as UTCTimestamp, value: drawing.pointA.price },
          { time: drawing.pointB.time as UTCTimestamp, value: drawing.pointB.price },
        ]);
        artifactsRef.current.push({ remove: () => chart.removeSeries(lineSeries) });
      }
    });

    return () => {
      artifactsRef.current.forEach((artifact) => artifact.remove());
      artifactsRef.current = [];
    };
  }, [chart, candleSeries, drawings]);

  function deleteLast() {
    persist(drawings.slice(0, -1));
  }

  function clearAll() {
    persist([]);
  }

  return (
    <div className="drawing-tools">
      <button
        type="button"
        className={`secondary-button${mode === "horizontal" ? " secondary-button--active" : ""}`}
        onClick={() => setMode(mode === "horizontal" ? "idle" : "horizontal")}
      >
        + Línea horizontal
      </button>
      <button
        type="button"
        className={`secondary-button${mode === "trendline-1" || mode === "trendline-2" ? " secondary-button--active" : ""}`}
        onClick={() => setMode(mode === "idle" ? "trendline-1" : "idle")}
      >
        + Línea de tendencia
      </button>
      <button type="button" className="secondary-button" onClick={deleteLast} disabled={drawings.length === 0}>
        Borrar última
      </button>
      <button type="button" className="secondary-button" onClick={clearAll} disabled={drawings.length === 0}>
        Limpiar todo
      </button>
      {mode === "horizontal" && <span className="view-note">Click en el gráfico para fijar el nivel…</span>}
      {mode === "trendline-1" && <span className="view-note">Click para el primer punto…</span>}
      {mode === "trendline-2" && <span className="view-note">Click para el segundo punto…</span>}
    </div>
  );
}
