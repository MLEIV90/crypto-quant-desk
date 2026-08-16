/**
 * Gráfico de líneas genérico y sincronizado (lightweight-charts), Fase 8c
 * — reutilizado por la vista de Riesgo (precio, volatilidad condicional
 * GARCH) y la de Backtest (curvas de equity). Mismo patrón de ciclo de
 * vida que `components/Chart.tsx` (crear el chart una vez, reconstruir
 * las series cuando cambian los datos) pero sin velas ni panes de
 * osciladores — solo N líneas en un único pane.
 */

import { useEffect, useRef } from "react";
import { ColorType, createChart, CrosshairMode, LineSeries } from "lightweight-charts";
import type { IChartApi, ISeriesApi, Time, UTCTimestamp } from "lightweight-charts";
import { COLORS } from "../theme";

export interface LineSeriesSpec {
  id: string;
  label: string;
  color: string;
  data: { time: string; value: number | null }[];
  lineWidth?: 1 | 2 | 3 | 4;
}

interface LineChartPanelProps {
  series: LineSeriesSpec[];
  height?: number;
}

function toTimestamp(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

export function LineChartPanel({ series, height = 320 }: LineChartPanelProps) {
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
      return lineSeries;
    });

    chart.timeScale().fitContent();
  }, [series]);

  return <div ref={containerRef} className="chart-container" style={{ height }} />;
}
