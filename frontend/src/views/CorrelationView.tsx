/**
 * Vista "Correlación" (Fase 13b, enriquecida en Fase 29) — consume
 * `/api/correlation` (sobre `eda.eda_report.correlation_matrix`, foto
 * estática de las 5 monedas) y `/api/correlation/rolling` (Fase 29, sobre
 * `eda.eda_report.rolling_pairwise_correlation`, correlación de UN par en
 * el tiempo).
 *
 * Fase 29: el heatmap solo mostraba una FOTO — el propio texto ya decía
 * que las correlaciones cambian con el tiempo, pero no lo mostraba. Ahora
 * hay (1) un gráfico de correlación rolling para un par elegido, con la
 * comparación contra su propio promedio histórico, (2) un texto que
 * traduce la matriz a "cuán diversificado está este universo" con un
 * índice resumen, y (3) las celdas más/menos correlacionadas resaltadas
 * directamente en el heatmap, no solo en tarjetas aparte.
 *
 * Reutiliza `PeriodSelector` (Fase 10a) para la ventana y el `interval`
 * compartido del header (igual que Estadística/Comparación/Arbitraje) —
 * el método (Pearson/Spearman) y el par elegido para la correlación
 * rolling son estado propio de esta vista.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getCorrelation, getCorrelationCsv, getCorrelationRolling } from "../api";
import { CorrelationHeatmap, type HeatmapHighlight } from "../components/CorrelationHeatmap";
import {
  CorrelationMethodSelector,
  DEFAULT_CORRELATION_METHOD,
  type CorrelationMethodKey,
} from "../components/CorrelationMethodSelector";
import { CsvDownloadButton } from "../components/CsvDownloadButton";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type ReferenceLineSpec } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { candleLimitForPeriod, DEFAULT_PERIOD, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { StatusMessage } from "../components/StatusMessage";
import {
  CORRELATION_DIVERSIFICATION_INDEX_HELP,
  CORRELATION_DIVERSIFICATION_INTRO,
  CORRELATION_EXTREME_PAIR_HELP,
  CORRELATION_INTRO_HELP,
  CORRELATION_ROLLING_HELP,
  CORRELATION_VS_HISTORICAL_HELP,
} from "../helpTexts";
import { COLORS } from "../theme";

const ROLLING_WINDOW = 90;
const DEFAULT_PAIR_A = "BTC";
const DEFAULT_PAIR_B = "ETH";

interface ExtremePair {
  a: string;
  b: string;
  valor: number;
}

function formatCorr(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

interface CorrelationViewProps {
  assets: string[];
  interval: string;
}

export function CorrelationView({ assets, interval }: CorrelationViewProps) {
  const [period, setPeriod] = useState<PeriodKey>(DEFAULT_PERIOD);
  const [method, setMethod] = useState<CorrelationMethodKey>(DEFAULT_CORRELATION_METHOD);
  // Fase 29: par elegido para el gráfico de correlación rolling — default
  // BTC-ETH, el par más correlacionado (ver ARBITRAGE_/COMPARISON_ views,
  // mismo patrón de dos <select> que ya usa ArbitrageView).
  const [pairA, setPairA] = useState(DEFAULT_PAIR_A);
  const [pairB, setPairB] = useState(DEFAULT_PAIR_B);
  const candleLimit = candleLimitForPeriod(period, interval);

  const correlationQuery = useQuery({
    queryKey: ["correlation", interval, candleLimit, method],
    queryFn: () => getCorrelation(interval, candleLimit, method),
  });

  const rollingQuery = useQuery({
    queryKey: ["correlation-rolling", pairA, pairB, interval, candleLimit],
    queryFn: () => getCorrelationRolling(pairA, pairB, interval, ROLLING_WINDOW, candleLimit),
    enabled: pairA !== pairB,
  });

  const correlation = correlationQuery.data;
  const error = correlationQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  const rolling = rollingQuery.data;
  const rollingError = rollingQuery.error;
  const rollingErrorMessage =
    rollingError instanceof ApiError ? rollingError.message : rollingError ? String(rollingError) : null;

  const extremes = useMemo(() => {
    if (!correlation) return null;
    const { activos, matriz } = correlation;
    let max: ExtremePair | null = null;
    let min: ExtremePair | null = null;
    for (let i = 0; i < activos.length; i++) {
      for (let j = i + 1; j < activos.length; j++) {
        const valor = matriz[i][j];
        if (valor === null) continue;
        if (!max || valor > max.valor) max = { a: activos[i], b: activos[j], valor };
        if (!min || valor < min.valor) min = { a: activos[i], b: activos[j], valor };
      }
    }
    return { max, min };
  }, [correlation]);

  // Fase 29 (mejora 4): resaltar en el heatmap mismo, no solo en tarjetas aparte.
  const heatmapHighlights: HeatmapHighlight[] = useMemo(() => {
    const highlights: HeatmapHighlight[] = [];
    if (extremes?.max) highlights.push({ a: extremes.max.a, b: extremes.max.b, kind: "max" });
    if (extremes?.min) highlights.push({ a: extremes.min.a, b: extremes.min.b, kind: "min" });
    return highlights;
  }, [extremes]);

  // Fase 29 (mejora 2): "diversificación" = 1 - correlación media entre
  // todos los pares (excluyendo la diagonal) — un resumen de una sola
  // línea de la misma matriz que ya se ve arriba.
  const diversification = useMemo(() => {
    if (!correlation) return null;
    const { activos, matriz } = correlation;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < activos.length; i++) {
      for (let j = i + 1; j < activos.length; j++) {
        const valor = matriz[i][j];
        if (valor === null) continue;
        sum += valor;
        count++;
      }
    }
    if (count === 0) return null;
    const media = sum / count;
    return { media, indice: 1 - media };
  }, [correlation]);

  function handlePairAChange(next: string) {
    setPairA(next);
    if (next === pairB) {
      const fallback = assets.find((a) => a !== next);
      if (fallback) setPairB(fallback);
    }
  }

  function handlePairBChange(next: string) {
    setPairB(next);
    if (next === pairA) {
      const fallback = assets.find((a) => a !== next);
      if (fallback) setPairA(fallback);
    }
  }

  const rollingSeries = useMemo(() => {
    if (!rolling) return [];
    return [
      {
        id: "rolling-corr",
        label: `Correlación ${rolling.asset_a}-${rolling.asset_b} (rolling ${rolling.window}d)`,
        color: COLORS.accent,
        data: rolling.fechas.map((fecha, i) => ({ time: fecha, value: rolling.correlacion[i] ?? null })),
      },
    ];
  }, [rolling]);

  const rollingReferenceLines: ReferenceLineSpec[] = useMemo(() => {
    if (!rolling) return [];
    const lines: ReferenceLineSpec[] = [
      { price: 1, label: "+1", color: COLORS.textMuted },
      { price: 0, label: "0", color: COLORS.textMuted },
    ];
    if (rolling.correlacion_promedio_historico !== null) {
      lines.push({
        price: rolling.correlacion_promedio_historico,
        label: "promedio histórico",
        color: COLORS.warning,
      });
    }
    return lines;
  }, [rolling]);

  return (
    <section className="view">
      <p className="view-note">{CORRELATION_INTRO_HELP}</p>

      <div className="chart-controls-row">
        <PeriodSelector active={period} onChange={setPeriod} />
        <CorrelationMethodSelector active={method} onChange={setMethod} />
        <CsvDownloadButton
          label="Descargar CSV"
          filename={`correlation_${interval}_${method}.csv`}
          fetchCsv={() => getCorrelationCsv(interval, candleLimit, method)}
          queryKey={["export-correlation-csv", interval, candleLimit, method]}
        />
      </div>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && correlationQuery.isLoading && (
        <StatusMessage kind="loading">Calculando correlaciones…</StatusMessage>
      )}

      {!errorMessage && correlation && (
        <>
          <CorrelationHeatmap
            activos={correlation.activos}
            matriz={correlation.matriz}
            highlights={heatmapHighlights}
          />

          <p className="view-note">
            {correlation.fechas_n} fechas comunes usadas ({correlation.method === "pearson" ? "Pearson" : "Spearman"}).
            Borde amarillo = par más correlacionado, borde celeste = par menos correlacionado.
          </p>

          {extremes?.max && extremes?.min && diversification && (
            <div className="metric-grid">
              <MetricCard
                label="Par más correlacionado"
                value={`${extremes.max.a}-${extremes.max.b} (${formatCorr(extremes.max.valor)})`}
                help={CORRELATION_EXTREME_PAIR_HELP.max}
              />
              <MetricCard
                label="Par menos correlacionado"
                value={`${extremes.min.a}-${extremes.min.b} (${formatCorr(extremes.min.valor)})`}
                help={CORRELATION_EXTREME_PAIR_HELP.min}
              />
              <MetricCard
                label="Diversificación del universo"
                value={diversification.indice.toFixed(2)}
                help={CORRELATION_DIVERSIFICATION_INDEX_HELP}
                subtext={`correlación media entre pares: ${formatCorr(diversification.media)}`}
              />
            </div>
          )}

          <p className="view-note">{CORRELATION_DIVERSIFICATION_INTRO}</p>

          <h3 className="panel-subtitle">
            Correlación en el tiempo (ventana móvil de {ROLLING_WINDOW} días)
            <InfoTooltip text={CORRELATION_ROLLING_HELP} />
          </h3>
          <div className="asset-selector">
            <label className="asset-selector__field">
              Moneda A
              <select value={pairA} onChange={(event) => handlePairAChange(event.target.value)}>
                {assets.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="asset-selector__field">
              Moneda B
              <select value={pairB} onChange={(event) => handlePairBChange(event.target.value)}>
                {assets.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {rollingErrorMessage && <StatusMessage kind="error">{rollingErrorMessage}</StatusMessage>}
          {!rollingErrorMessage && rollingQuery.isLoading && (
            <StatusMessage kind="loading">
              Calculando correlación rolling de {pairA}-{pairB}…
            </StatusMessage>
          )}

          {!rollingErrorMessage && rolling && (
            <>
              <LineChartPanel series={rollingSeries} height={320} referenceLines={rollingReferenceLines} />
              <p className="view-note">
                Cuando la correlación se dispara hacia 1, la diversificación entre estas dos monedas
                desaparece — suele pasar en las crisis, justo cuando más la necesitás.
              </p>
              {rolling.correlacion_actual !== null && rolling.correlacion_promedio_historico !== null && (
                <p className="view-note">
                  Correlación actual: <strong>{formatCorr(rolling.correlacion_actual)}</strong> —{" "}
                  {rolling.correlacion_actual > rolling.correlacion_promedio_historico
                    ? "por ENCIMA"
                    : "por DEBAJO"}{" "}
                  del promedio histórico ({formatCorr(rolling.correlacion_promedio_historico)}):{" "}
                  {rolling.correlacion_actual > rolling.correlacion_promedio_historico
                    ? "estas dos monedas se están moviendo MÁS juntas de lo habitual."
                    : "estas dos monedas se están moviendo más independiente de lo habitual."}
                  <InfoTooltip text={CORRELATION_VS_HISTORICAL_HELP} placement="bottom" />
                </p>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
