/**
 * Gráfico de líneas genérico y sincronizado (lightweight-charts), Fase 8c
 * — reutilizado por la vista de Riesgo (precio, volatilidad condicional
 * GARCH) y la de Backtest (curvas de equity). Mismo patrón de ciclo de
 * vida que `components/Chart.tsx` (crear el chart una vez, reconstruir
 * las series cuando cambian los datos) pero sin velas ni panes de
 * osciladores — solo N líneas en un único pane.
 *
 * `referenceLines` (Fase 12b, vista "Arbitraje"): líneas horizontales
 * opcionales (`series.createPriceLine()`, misma primitiva que ya usa
 * `Chart.tsx` para Fibonacci/soporte-resistencia/pivotes) ancladas a la
 * PRIMERA serie — pensadas para las bandas de z-score (+2/-2/0), pero
 * genéricas para cualquier gráfico de línea futuro que las necesite.
 *
 * `markers` por serie (Fase 15b, vista "Arbitraje"): puntos puntuales
 * marcados sobre ESA serie (`lightweight-charts.createSeriesMarkers`, la
 * primitiva nativa de anotaciones de la librería) — pensado para resaltar
 * los extremos históricos del z-score (|z|>=2), con posición EXACTA en el
 * eje de precio (`position: "atPriceMiddle"` + `price`), no relativa a la
 * vela como "aboveBar"/"belowBar".
 */

import { useEffect, useRef } from "react";
import { ColorType, createChart, createSeriesMarkers, CrosshairMode, LineSeries, LineStyle } from "lightweight-charts";
import type { IChartApi, ISeriesApi, Time, UTCTimestamp } from "lightweight-charts";
import { COLORS } from "../theme";

export interface LineSeriesMarkerSpec {
  time: string;
  price: number;
  color: string;
  shape?: "circle" | "square" | "arrowUp" | "arrowDown";
  text?: string;
}

export interface LineSeriesSpec {
  id: string;
  label: string;
  color: string;
  data: { time: string; value: number | null }[];
  lineWidth?: 1 | 2 | 3 | 4;
  markers?: LineSeriesMarkerSpec[];
}

export interface ReferenceLineSpec {
  price: number;
  label?: string;
  color?: string;
}

interface LineChartPanelProps {
  series: LineSeriesSpec[];
  height?: number;
  referenceLines?: ReferenceLineSpec[];
}

function toTimestamp(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

export function LineChartPanel({ series, height = 320, referenceLines = [] }: LineChartPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesApiRef = useRef<ISeriesApi<"Line", Time>[]>([]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      layout: { background: { type: ColorType.Solid, color: COLORS.background }, textColor: COLORS.textMuted },
      grid: { vertLines: { color: COLORS.border }, horzLines: { color: COLORS.border } },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: { borderColor: COLORS.border, timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: COLORS.border },
      autoSize: true,
    });
    chartRef.current = chart;

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesApiRef.current = [];
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    seriesApiRef.current.forEach((api) => chart.removeSeries(api));
    seriesApiRef.current = series.map((spec) => {
      const lineSeries = chart.addSeries(LineSeries, {
        color: spec.color,
        lineWidth: spec.lineWidth ?? 2,
        title: spec.label,
        priceLineVisible: false,
      });
      const points: { time: UTCTimestamp; value: number }[] = [];
      spec.data.forEach((point) => {
        if (point.value !== null && point.value !== undefined) {
          points.push({ time: toTimestamp(point.time), value: point.value });
        }
      });
      lineSeries.setData(points);

      if (spec.markers && spec.markers.length > 0) {
        createSeriesMarkers(
          lineSeries,
          spec.markers.map((marker) => ({
            time: toTimestamp(marker.time),
            position: "atPriceMiddle" as const,
            price: marker.price,
            color: marker.color,
            shape: marker.shape ?? "circle",
            text: marker.text,
          })),
        );
      }

      return lineSeries;
    });

    const anchor = seriesApiRef.current[0];
    if (anchor) {
      referenceLines.forEach((ref) => {
        anchor.createPriceLine({
          price: ref.price,
          color: ref.color ?? COLORS.textMuted,
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: true,
          title: ref.label ?? "",
        });
      });
    }

    chart.timeScale().fitContent();
  }, [series, referenceLines]);

  return <div ref={containerRef} className="chart-container" style={{ height }} />;
}
