/**
 * Gráfico principal: velas japonesas (lightweight-charts, TradingView) con
 * zoom/paneo/crosshair NATIVOS de la librería, overlays toggleables sobre
 * el precio y osciladores en panes propios sincronizados en el eje
 * temporal. Fase 8b.
 *
 * Solo dibuja lo que ya viene de `/api/ohlcv`/`/api/studies` (Fase 8a) —
 * no calcula ningún indicador acá, misma regla de separación modelo/vista
 * que la app de escritorio (ver `app/widgets/technical_chart.py`, su
 * equivalente en PySide6/matplotlib).
 *
 * Una única instancia de `IChartApi` vive durante todo el ciclo de vida
 * del componente (se crea una vez en un `useEffect` con `[]`, se destruye
 * en el cleanup) — los panes de los osciladores tienen que ser panes de
 * ESA MISMA instancia para quedar sincronizados en el eje temporal con las
 * velas; no son gráficos aparte.
 */

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
} from "lightweight-charts";
import type { IChartApi, IPaneApi, ISeriesApi, Time, UTCTimestamp } from "lightweight-charts";
import type { OHLCVResponse, StudiesResponse } from "../types";
import type { OverlayKey } from "./StudyToggles";
import type { OscillatorKey } from "./OscillatorPanel";
import { COLORS } from "../theme";

interface ChartProps {
  ohlcv: OHLCVResponse;
  studies: StudiesResponse;
  activeOverlays: Set<OverlayKey>;
  activeOscillators: Set<OscillatorKey>;
}

type LineSeriesApi = ISeriesApi<"Line", Time>;

interface OverlayArtifact {
  series: LineSeriesApi[];
  priceLines: { series: ISeriesApi<"Candlestick", Time>; line: ReturnType<LineSeriesApi["createPriceLine"]> }[];
}

interface OscillatorArtifact {
  pane: IPaneApi<Time>;
}

function toTimestamp(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

function toLineData(fechas: string[], values: (number | null)[]): { time: UTCTimestamp; value: number }[] {
  const points: { time: UTCTimestamp; value: number }[] = [];
  fechas.forEach((fecha, index) => {
    const value = values[index];
    if (value !== null && value !== undefined) {
      points.push({ time: toTimestamp(fecha), value });
    }
  });
  return points;
}

const REFERENCE_LINE_STYLE = { color: COLORS.textMuted, lineWidth: 1 as const, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: "" };

function addReferenceLine(series: LineSeriesApi, price: number): void {
  series.createPriceLine({ price, ...REFERENCE_LINE_STYLE });
}

function createOverlay(
  chart: IChartApi,
  candleSeries: ISeriesApi<"Candlestick", Time>,
  key: OverlayKey,
  studies: StudiesResponse,
): OverlayArtifact {
  const empty: OverlayArtifact = { series: [], priceLines: [] };

  switch (key) {
    case "sma20":
    case "sma50":
    case "ema12":
    case "ema26": {
      const fieldByKey: Record<typeof key, keyof StudiesResponse> = {
        sma20: "sma_20",
        sma50: "sma_50",
        ema12: "ema_12",
        ema26: "ema_26",
      } as const;
      const colorByKey: Record<typeof key, string> = {
        sma20: COLORS.sma20,
        sma50: COLORS.sma50,
        ema12: COLORS.ema12,
        ema26: COLORS.ema26,
      } as const;
      const titleByKey: Record<typeof key, string> = {
        sma20: "SMA 20",
        sma50: "SMA 50",
        ema12: "EMA 12",
        ema26: "EMA 26",
      } as const;
      const series = chart.addSeries(LineSeries, {
        color: colorByKey[key],
        lineWidth: 2,
        title: titleByKey[key],
        priceLineVisible: false,
        lastValueVisible: false,
      });
      series.setData(toLineData(studies.fechas, studies[fieldByKey[key]] as (number | null)[]));
      return { series: [series], priceLines: [] };
    }
    case "bollinger": {
      const upper = chart.addSeries(LineSeries, {
        color: COLORS.bollinger, lineWidth: 1, lineStyle: LineStyle.Dashed, title: "BB superior",
        priceLineVisible: false, lastValueVisible: false,
      });
      const mid = chart.addSeries(LineSeries, {
        color: COLORS.bollinger, lineWidth: 1, lineStyle: LineStyle.Dotted, title: "BB media",
        priceLineVisible: false, lastValueVisible: false,
      });
      const lower = chart.addSeries(LineSeries, {
        color: COLORS.bollinger, lineWidth: 1, lineStyle: LineStyle.Dashed, title: "BB inferior",
        priceLineVisible: false, lastValueVisible: false,
      });
      upper.setData(toLineData(studies.fechas, studies.bb_upper));
      mid.setData(toLineData(studies.fechas, studies.bb_mid));
      lower.setData(toLineData(studies.fechas, studies.bb_lower));
      return { series: [upper, mid, lower], priceLines: [] };
    }
    case "fibonacci": {
      if (!studies.fibonacci) return empty;
      const priceLines = Object.entries(studies.fibonacci).map(([label, level]) => ({
        series: candleSeries,
        line: candleSeries.createPriceLine({
          price: level, color: COLORS.fibonacci, lineWidth: 1, lineStyle: LineStyle.Dotted,
          axisLabelVisible: true, title: `fib ${label}`,
        }),
      }));
      return { series: [], priceLines };
    }
    case "supportResistance": {
      const { resistencia, soporte } = studies.soporte_resistencia;
      const priceLines = [
        ...resistencia.map((level) => ({
          series: candleSeries,
          line: candleSeries.createPriceLine({
            price: level, color: COLORS.resistance, lineWidth: 1, lineStyle: LineStyle.Solid,
            axisLabelVisible: true, title: "R",
          }),
        })),
        ...soporte.map((level) => ({
          series: candleSeries,
          line: candleSeries.createPriceLine({
            price: level, color: COLORS.support, lineWidth: 1, lineStyle: LineStyle.Solid,
            axisLabelVisible: true, title: "S",
          }),
        })),
      ];
      return { series: [], priceLines };
    }
    case "pivots": {
      const pivotKeys = ["P", "R1", "S1"] as const;
      const priceLines = pivotKeys
        .filter((pivotKey) => studies.pivotes[pivotKey] !== undefined)
        .map((pivotKey) => ({
          series: candleSeries,
          line: candleSeries.createPriceLine({
            price: studies.pivotes[pivotKey], color: COLORS.pivot, lineWidth: 1, lineStyle: LineStyle.LargeDashed,
            axisLabelVisible: true, title: pivotKey,
          }),
        }));
      return { series: [], priceLines };
    }
    default:
      return empty;
  }
}

function removeOverlay(chart: IChartApi, artifact: OverlayArtifact): void {
  artifact.series.forEach((series) => chart.removeSeries(series));
  artifact.priceLines.forEach(({ series, line }) => series.removePriceLine(line));
}

function createOscillator(chart: IChartApi, key: OscillatorKey, studies: StudiesResponse): OscillatorArtifact {
  const pane = chart.addPane();

  if (key === "rsi") {
    const rsi = pane.addSeries(LineSeries, {
      color: COLORS.rsi, lineWidth: 2, title: "RSI 14", priceLineVisible: false, lastValueVisible: false,
    });
    rsi.setData(toLineData(studies.fechas, studies.rsi_14));
    addReferenceLine(rsi, 70);
    addReferenceLine(rsi, 30);
  } else if (key === "macd") {
    const macdLine = pane.addSeries(LineSeries, {
      color: COLORS.macdLine, lineWidth: 2, title: "MACD", priceLineVisible: false, lastValueVisible: false,
    });
    macdLine.setData(toLineData(studies.fechas, studies.macd));
    const signalLine = pane.addSeries(LineSeries, {
      color: COLORS.macdSignal, lineWidth: 1, title: "Señal", priceLineVisible: false, lastValueVisible: false,
    });
    signalLine.setData(toLineData(studies.fechas, studies.macd_signal));
    const histogram = pane.addSeries(HistogramSeries, { title: "Hist.", priceLineVisible: false, lastValueVisible: false });
    const histogramData: { time: UTCTimestamp; value: number; color: string }[] = [];
    studies.fechas.forEach((fecha, index) => {
      const value = studies.macd_hist[index];
      if (value !== null && value !== undefined) {
        histogramData.push({
          time: toTimestamp(fecha), value, color: value >= 0 ? COLORS.candleUp : COLORS.candleDown,
        });
      }
    });
    histogram.setData(histogramData);
  } else if (key === "stochastic") {
    const kLine = pane.addSeries(LineSeries, {
      color: COLORS.stochK, lineWidth: 2, title: "%K", priceLineVisible: false, lastValueVisible: false,
    });
    kLine.setData(toLineData(studies.fechas, studies.stoch_k));
    const dLine = pane.addSeries(LineSeries, {
      color: COLORS.stochD, lineWidth: 1, title: "%D", priceLineVisible: false, lastValueVisible: false,
    });
    dLine.setData(toLineData(studies.fechas, studies.stoch_d));
    addReferenceLine(kLine, 80);
    addReferenceLine(kLine, 20);
  }

  return { pane };
}

function removeOscillator(chart: IChartApi, artifact: OscillatorArtifact): void {
  chart.removePane(artifact.pane.paneIndex());
}

export function Chart({ ohlcv, studies, activeOverlays, activeOscillators }: ChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick", Time> | null>(null);
  const overlayArtifactsRef = useRef<Map<OverlayKey, OverlayArtifact>>(new Map());
  const oscillatorArtifactsRef = useRef<Map<OscillatorKey, OscillatorArtifact>>(new Map());

  // El chart se crea UNA sola vez y se destruye al desmontar el componente.
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
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: COLORS.candleUp,
      downColor: COLORS.candleDown,
      borderUpColor: COLORS.candleUp,
      borderDownColor: COLORS.candleDown,
      wickUpColor: COLORS.candleUp,
      wickDownColor: COLORS.candleDown,
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    return () => {
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      overlayArtifactsRef.current.clear();
      oscillatorArtifactsRef.current.clear();
    };
  }, []);

  // Velas: se recargan cuando cambia el activo/timeframe/ventana.
  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    if (!candleSeries) return;
    candleSeries.setData(
      ohlcv.velas.map((vela) => ({
        time: toTimestamp(vela.fecha),
        open: vela.open,
        high: vela.high,
        low: vela.low,
        close: vela.close,
      })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [ohlcv]);

  // Overlays: se reconstruyen ENTEROS cuando cambia qué overlays están
  // activos o cuando cambian los datos de estudios (nuevo activo/
  // timeframe) — simple y suficientemente barato (a lo sumo unas pocas
  // series de un puñado de cientos de puntos) para no necesitar un diff
  // más fino que arriesgue dejar una serie con datos viejos.
  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries) return;

    overlayArtifactsRef.current.forEach((artifact) => removeOverlay(chart, artifact));
    overlayArtifactsRef.current.clear();
    activeOverlays.forEach((key) => {
      overlayArtifactsRef.current.set(key, createOverlay(chart, candleSeries, key, studies));
    });
  }, [activeOverlays, studies]);

  // Osciladores: mismo criterio (reconstrucción completa) que los overlays.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    oscillatorArtifactsRef.current.forEach((artifact) => removeOscillator(chart, artifact));
    oscillatorArtifactsRef.current.clear();
    activeOscillators.forEach((key) => {
      oscillatorArtifactsRef.current.set(key, createOscillator(chart, key, studies));
    });
  }, [activeOscillators, studies]);

  return <div ref={containerRef} className="chart-container" />;
}
