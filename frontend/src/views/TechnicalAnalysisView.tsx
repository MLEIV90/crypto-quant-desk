/**
 * Vista "Análisis Técnico" (Fase 8b, extendida en 8c con el panel del
 * sugeridor y en 8d con alertas + dibujo). Chart + toggles vienen de 8b
 * tal cual; `SuggesterPanel` es de 8c. Lo nuevo en 8d: `AlertsPanel`
 * (reglas client-side sobre `/api/studies`) y `DrawingTools` (líneas sobre
 * el gráfico) — `Chart` expone su `IChartApi`/serie de velas vía
 * `onChartReady` para que `DrawingTools` dibuje sobre EL MISMO chart.
 */

import { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { IChartApi, ISeriesApi, Time } from "lightweight-charts";
import { ApiError, getOhlcv, getStudies, getSuggester } from "../api";
import { AlertsPanel } from "../components/AlertsPanel";
import { Chart } from "../components/Chart";
import { DrawingTools } from "../components/DrawingTools";
import { DEFAULT_ACTIVE_OSCILLATORS, OscillatorPanel, type OscillatorKey } from "../components/OscillatorPanel";
import { DEFAULT_ACTIVE_OVERLAYS, StudyToggles, type OverlayKey } from "../components/StudyToggles";
import { SuggesterPanel } from "../components/SuggesterPanel";
import { StatusMessage } from "../components/StatusMessage";

const CANDLE_LIMIT = 300;

function toggleInSet<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

interface TechnicalAnalysisViewProps {
  asset: string;
  interval: string;
}

export function TechnicalAnalysisView({ asset, interval }: TechnicalAnalysisViewProps) {
  const [activeOverlays, setActiveOverlays] = useState<Set<OverlayKey>>(new Set(DEFAULT_ACTIVE_OVERLAYS));
  const [activeOscillators, setActiveOscillators] = useState<Set<OscillatorKey>>(
    new Set(DEFAULT_ACTIVE_OSCILLATORS),
  );
  const [chartApi, setChartApi] = useState<{
    chart: IChartApi;
    series: ISeriesApi<"Candlestick", Time>;
  } | null>(null);

  const handleChartReady = useCallback(
    (chart: IChartApi | null, series: ISeriesApi<"Candlestick", Time> | null) => {
      setChartApi(chart && series ? { chart, series } : null);
    },
    [],
  );

  const ohlcvQuery = useQuery({
    queryKey: ["ohlcv", asset, interval],
    queryFn: () => getOhlcv(asset, interval, CANDLE_LIMIT),
  });
  const studiesQuery = useQuery({
    queryKey: ["studies", asset, interval],
    queryFn: () => getStudies(asset, interval, CANDLE_LIMIT),
  });
  const suggesterQuery = useQuery({
    queryKey: ["suggester", asset, interval],
    queryFn: () => getSuggester(asset, interval),
  });

  const ohlcv = ohlcvQuery.data;
  const studies = studiesQuery.data;
  const suggester = suggesterQuery.data;

  const isLoading = ohlcvQuery.isLoading || studiesQuery.isLoading;
  const error = ohlcvQuery.error ?? studiesQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <div className="toggles-row">
        <StudyToggles active={activeOverlays} onToggle={(key) => setActiveOverlays((prev) => toggleInSet(prev, key))} />
        <OscillatorPanel
          active={activeOscillators}
          onToggle={(key) => setActiveOscillators((prev) => toggleInSet(prev, key))}
        />
      </div>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && isLoading && <StatusMessage kind="loading">Cargando {asset}…</StatusMessage>}

      {!errorMessage && ohlcv && studies && (
        <div className="technical-layout">
          <div className="technical-layout__chart-col">
            <DrawingTools
              asset={asset}
              interval={interval}
              chart={chartApi?.chart ?? null}
              candleSeries={chartApi?.series ?? null}
            />
            <Chart
              ohlcv={ohlcv}
              studies={studies}
              activeOverlays={activeOverlays}
              activeOscillators={activeOscillators}
              onChartReady={handleChartReady}
            />
          </div>
          <div className="technical-layout__side-col">
            {suggester ? (
              <SuggesterPanel data={suggester} />
            ) : (
              <aside className="suggester-panel">
                {suggesterQuery.isLoading && <StatusMessage kind="loading">Calculando sugerencia…</StatusMessage>}
                {suggesterQuery.isError && (
                  <StatusMessage kind="error">No se pudo calcular la sugerencia de consenso.</StatusMessage>
                )}
              </aside>
            )}
            <AlertsPanel asset={asset} interval={interval} studies={studies} ohlcv={ohlcv} />
          </div>
        </div>
      )}
    </section>
  );
}
