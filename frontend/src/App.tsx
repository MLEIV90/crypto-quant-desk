/**
 * Punto de entrada de la UI (Fase 8b, navegación multi-vista agregada en
 * 8c): estado compartido de activo/timeframe + qué vista está activa,
 * disclaimer global siempre visible. Cada vista hace su propio fetch vía
 * React Query (ver `views/`) — acá solo vive la navegación y el estado
 * que de verdad se comparte entre vistas.
 */

import { useMemo, useState } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import "./App.css";
import { getAssets } from "./api";
import { AssetSelector } from "./components/AssetSelector";
import { DataStatusBar } from "./components/DataStatusBar";
import { NavTabs, type ViewKey } from "./components/NavTabs";
import { WatchlistPanel } from "./components/WatchlistPanel";
import { TechnicalAnalysisView } from "./views/TechnicalAnalysisView";
import { RiskView } from "./views/RiskView";
import { BacktestView } from "./views/BacktestView";
import { ResearchView } from "./views/ResearchView";

const DEFAULT_ASSET = "BTC";
const DEFAULT_INTERVAL = "1d";

const DISCLAIMER_TEXT =
  "Herramienta de análisis y gestión de riesgo. No es asesoramiento financiero ni predicción de precio.";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 60_000 } },
});

function Dashboard() {
  const [activeView, setActiveView] = useState<ViewKey>("tecnico");
  const [asset, setAsset] = useState(DEFAULT_ASSET);
  const [interval, setInterval] = useState(DEFAULT_INTERVAL);

  const assetsQuery = useQuery({ queryKey: ["assets"], queryFn: getAssets });
  const assets = useMemo(() => assetsQuery.data?.activos ?? [asset], [assetsQuery.data, asset]);
  const timeframes = useMemo(() => assetsQuery.data?.timeframes ?? [interval], [assetsQuery.data, interval]);

  // Riesgo/Backtest/Research siempre operan en diario (DEFAULT_RISK_INTERVAL
  // en api/main.py) — el selector de timeframe ni se muestra ahí (ver
  // AssetSelector.showTimeframe), así que el estado de datos que corresponde
  // mostrar en esas vistas es siempre el de "1d", no el de `interval`.
  const dataStatusInterval = activeView === "tecnico" ? interval : "1d";

  return (
    <div className="app">
      <header className="app__header">
        <h1>crypto-quant-desk</h1>
        <NavTabs active={activeView} onChange={setActiveView} />
        <AssetSelector
          assets={assets}
          timeframes={timeframes}
          asset={asset}
          timeframe={interval}
          onAssetChange={setAsset}
          onTimeframeChange={setInterval}
          showTimeframe={activeView === "tecnico"}
        />
      </header>

      <WatchlistPanel assets={assets} activeAsset={asset} onSelectAsset={setAsset} />

      <DataStatusBar asset={asset} interval={dataStatusInterval} />

      <p className="disclaimer">{DISCLAIMER_TEXT}</p>

      {activeView === "tecnico" && <TechnicalAnalysisView asset={asset} interval={interval} />}
      {activeView === "riesgo" && <RiskView asset={asset} />}
      {activeView === "backtest" && <BacktestView asset={asset} />}
      {activeView === "research" && <ResearchView asset={asset} />}
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
