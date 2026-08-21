/**
 * Vista "Análisis Técnico" (Fase 8b, extendida en 8c con el panel del
 * sugeridor, en 8d con alertas + dibujo, y en 10a con el selector de
 * período). Chart + toggles vienen de 8b tal cual; `SuggesterPanel` es de
 * 8c. Fase 8d: `AlertsPanel` (reglas client-side sobre `/api/studies`) y
 * `DrawingTools` (líneas sobre el gráfico) — `Chart` expone su
 * `IChartApi`/serie de velas vía `onChartReady` para que `DrawingTools`
 * dibuje sobre EL MISMO chart.
 *
 * Fase 10a: `CANDLE_LIMIT` fijo (300 velas, ~10 meses en diario) se
 * reemplaza por `PeriodSelector` — el mismo `candleLimit` calculado se le
 * pasa a `/api/ohlcv` Y `/api/studies`, así el precio y los osciladores
 * (RSI/MACD/Estocástico) siempre grafican exactamente la misma ventana de
 * velas y no se desincronizan entre sí al cambiar de período.
 *
 * Fase 10b: `ChartTypeSelector` (velas / Heikin-Ashi) al lado del período.
 *
 * Fase 13c: `AlertsPanel` ahora puede crear reglas para CUALQUIER moneda
 * (recibe `assets`) y avisa hacia arriba (`onAlertTriggered`) cuándo se
 * disparó una, para que `App.tsx` resalte esa moneda en el watchlist aunque
 * el usuario esté mirando otro activo. El volume profile (Fase 13a) ahora
 * se pide SIEMPRE (no solo con el toggle activo) porque las alertas de
 * POC/Value Area lo necesitan aunque el overlay esté apagado — es un
 * cálculo liviano (numpy vectorizado), a diferencia de /api/risk o
 * /api/prediction, así que no hace falta gatearlo.
 *
 * Fase 14 (F-04): con "Todo" + horario (~58.000 velas), `Chart` submuestrea
 * el DIBUJO por defecto (ver `../downsample.ts`). El checkbox "Vista
 * completa" de acá abajo (solo visible por encima del umbral) lo desactiva
 * para quien prefiera resolución total a costa de más lag de render.
 */

import { useCallback, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { IChartApi, ISeriesApi, Time } from "lightweight-charts";
import { ApiError, getOhlcv, getOhlcvCsv, getStudies, getSuggester, getVolumeProfile } from "../api";
import { AlertsPanel } from "../components/AlertsPanel";
import { Chart } from "../components/Chart";
import { ChartTypeSelector, DEFAULT_CHART_TYPE, type ChartTypeKey } from "../components/ChartTypeSelector";
import { CsvDownloadButton } from "../components/CsvDownloadButton";
import { DrawingTools } from "../components/DrawingTools";
import { DOWNSAMPLE_THRESHOLD } from "../downsample";
import { DEFAULT_ACTIVE_OSCILLATORS, OscillatorPanel, type OscillatorKey } from "../components/OscillatorPanel";
import { candleLimitForPeriod, DEFAULT_PERIOD, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { DEFAULT_ACTIVE_OVERLAYS, StudyToggles, type OverlayKey } from "../components/StudyToggles";
import { SuggesterPanel } from "../components/SuggesterPanel";
import { StatusMessage } from "../components/StatusMessage";

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
  assets: string[];
  onAlertTriggered?: (asset: string) => void;
}

export function TechnicalAnalysisView({ asset, interval, assets, onAlertTriggered }: TechnicalAnalysisViewProps) {
  const [activeOverlays, setActiveOverlays] = useState<Set<OverlayKey>>(new Set(DEFAULT_ACTIVE_OVERLAYS));
  const [activeOscillators, setActiveOscillators] = useState<Set<OscillatorKey>>(
    new Set(DEFAULT_ACTIVE_OSCILLATORS),
  );
  const [period, setPeriod] = useState<PeriodKey>(DEFAULT_PERIOD);
  const candleLimit = candleLimitForPeriod(period, interval);
  const [chartType, setChartType] = useState<ChartTypeKey>(DEFAULT_CHART_TYPE);
  const [disableDownsample, setDisableDownsample] = useState(false);
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
    queryKey: ["ohlcv", asset, interval, candleLimit],
    queryFn: () => getOhlcv(asset, interval, candleLimit),
  });
  const studiesQuery = useQuery({
    queryKey: ["studies", asset, interval, candleLimit],
    queryFn: () => getStudies(asset, interval, candleLimit),
  });
  const suggesterQuery = useQuery({
    queryKey: ["suggester", asset, interval],
    queryFn: () => getSuggester(asset, interval),
  });
  // Siempre se pide (Fase 13c: antes solo con el toggle activo, ver
  // docstring del componente) — mismo candleLimit que ohlcv/studies para
  // que el perfil respete el período elegido (PeriodSelector).
  const volumeProfileQuery = useQuery({
    queryKey: ["volume-profile", asset, interval, candleLimit],
    queryFn: () => getVolumeProfile(asset, interval, candleLimit),
  });

  const ohlcv = ohlcvQuery.data;
  const studies = studiesQuery.data;
  const suggester = suggesterQuery.data;
  const volumeProfile = volumeProfileQuery.data;

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
            <div className="chart-controls-row">
              <ChartTypeSelector active={chartType} onChange={setChartType} />
              <PeriodSelector active={period} onChange={setPeriod} />
              {ohlcv.velas.length > DOWNSAMPLE_THRESHOLD && (
                <label className="toggle-chip">
                  <input
                    type="checkbox"
                    checked={disableDownsample}
                    onChange={(event) => setDisableDownsample(event.target.checked)}
                  />
                  Vista completa ({ohlcv.velas.length.toLocaleString("es-AR")} velas, más lento)
                </label>
              )}
              <CsvDownloadButton
                label="Descargar CSV"
                filename={`ohlcv_${asset}_${interval}.csv`}
                fetchCsv={() => getOhlcvCsv(asset, interval, candleLimit)}
                queryKey={["export-ohlcv-csv", asset, interval, candleLimit]}
              />
            </div>
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
              chartType={chartType}
              volumeProfile={volumeProfile ?? null}
              disableDownsample={disableDownsample}
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
            <AlertsPanel
              asset={asset}
              interval={interval}
              assets={assets}
              studies={studies}
              ohlcv={ohlcv}
              volumeProfile={volumeProfile ?? null}
              onAlertTriggered={onAlertTriggered}
            />
          </div>
        </div>
      )}
    </section>
  );
}
