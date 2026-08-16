/**
 * Punto de entrada de la UI (Fase 8b): selector de activo/timeframe +
 * gráfico interactivo + toggles de overlays/osciladores. Consume la API
 * REST (Fase 8a) vía React Query — maneja carga/error acá, `Chart.tsx`
 * solo dibuja datos ya resueltos.
 *
 * Fuera de alcance de esta fase (quedan para 8c/8d): pestañas de riesgo/
 * backtest/sugeridor, alertas, dibujo manual sobre el gráfico. Estética
 * mínima a propósito — el pulido visual completo es Fase 8c.
 */

import { useMemo, useState } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import "./App.css";
import { ApiError, getAssets, getOhlcv, getStudies } from "./api";
import { AssetSelector } from "./components/AssetSelector";
import { Chart } from "./components/Chart";
import { StudyToggles, DEFAULT_ACTIVE_OVERLAYS, type OverlayKey } from "./components/StudyToggles";
import { OscillatorPanel, DEFAULT_ACTIVE_OSCILLATORS, type OscillatorKey } from "./components/OscillatorPanel";

const DEFAULT_ASSET = "BTC";
const DEFAULT_INTERVAL = "1d";
const CANDLE_LIMIT = 300;

const DISCLAIMER_TEXT =
  "Herramienta de análisis técnico. No es asesoramiento financiero ni predicción de precio.";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 60_000 } },
});

function toggleInSet<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function Dashboard() {
  const [asset, setAsset] = useState(DEFAULT_ASSET);
  const [interval, setInterval] = useState(DEFAULT_INTERVAL);
  const [activeOverlays, setActiveOverlays] = useState<Set<OverlayKey>>(new Set(DEFAULT_ACTIVE_OVERLAYS));
  const [activeOscillators, setActiveOscillators] = useState<Set<OscillatorKey>>(
    new Set(DEFAULT_ACTIVE_OSCILLATORS),
  );

  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: getAssets });
  const ohlcvQuery = useQuery({
    queryKey: ["ohlcv", asset, interval],
    queryFn: () => getOhlcv(asset, interval, CANDLE_LIMIT),
  });
  const studiesQuery = useQuery({
    queryKey: ["studies", asset, interval],
    queryFn: () => getStudies(asset, interval, CANDLE_LIMIT),
  });

  const assets = useMemo(() => assetsQuery.data?.activos ?? [asset], [assetsQuery.data, asset]);
  const timeframes = useMemo(
    () => assetsQuery.data?.timeframes ?? [interval],
    [assetsQuery.data, interval],
  );

  const isLoading = ohlcvQuery.isLoading || studiesQuery.isLoading;
  const error = ohlcvQuery.error ?? studiesQuery.error ?? assetsQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <div className="app">
      <header className="app__header">
        <h1>crypto-quant-desk</h1>
        <AssetSelector
          assets={assets}
          timeframes={timeframes}
          asset={asset}
          timeframe={interval}
          onAssetChange={setAsset}
          onTimeframeChange={setInterval}
        />
      </header>

      <p className="disclaimer">{DISCLAIMER_TEXT}</p>

      <div className="toggles-row">
        <StudyToggles active={activeOverlays} onToggle={(key) => setActiveOverlays((prev) => toggleInSet(prev, key))} />
        <OscillatorPanel
          active={activeOscillators}
          onToggle={(key) => setActiveOscillators((prev) => toggleInSet(prev, key))}
        />
      </div>

      {errorMessage && <div className="status status--error">{errorMessage}</div>}
      {!errorMessage && isLoading && <div className="status">Cargando {asset}…</div>}
      {!errorMessage && ohlcvQuery.data && studiesQuery.data && (
        <Chart
          ohlcv={ohlcvQuery.data}
          studies={studiesQuery.data}
          activeOverlays={activeOverlays}
          activeOscillators={activeOscillators}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}
