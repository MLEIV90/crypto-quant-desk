/**
 * Vista "Correlación" (Fase 13b) — consume `/api/correlation` (sobre
 * `eda.eda_report.correlation_matrix`, reutilizado tal cual). Mapa de
 * calor 5x5 de correlación de RETORNOS (no precios) entre las monedas de
 * `config.UNIVERSE`, sobre el período/intervalo/método elegidos.
 *
 * Reutiliza `PeriodSelector` (Fase 10a) para la ventana y el `interval`
 * compartido del header (igual que Estadística/Comparación/Arbitraje) —
 * el método (Pearson/Spearman) es estado propio de esta vista.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getCorrelation } from "../api";
import { CorrelationHeatmap } from "../components/CorrelationHeatmap";
import {
  CorrelationMethodSelector,
  DEFAULT_CORRELATION_METHOD,
  type CorrelationMethodKey,
} from "../components/CorrelationMethodSelector";
import { candleLimitForPeriod, DEFAULT_PERIOD, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { StatusMessage } from "../components/StatusMessage";
import { CORRELATION_INTRO_HELP } from "../helpTexts";

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
  interval: string;
}

export function CorrelationView({ interval }: CorrelationViewProps) {
  const [period, setPeriod] = useState<PeriodKey>(DEFAULT_PERIOD);
  const [method, setMethod] = useState<CorrelationMethodKey>(DEFAULT_CORRELATION_METHOD);
  const candleLimit = candleLimitForPeriod(period, interval);

  const correlationQuery = useQuery({
    queryKey: ["correlation", interval, candleLimit, method],
    queryFn: () => getCorrelation(interval, candleLimit, method),
  });

  const correlation = correlationQuery.data;
  const error = correlationQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

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

  return (
    <section className="view">
      <p className="view-note">{CORRELATION_INTRO_HELP}</p>

      <div className="chart-controls-row">
        <PeriodSelector active={period} onChange={setPeriod} />
        <CorrelationMethodSelector active={method} onChange={setMethod} />
      </div>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && correlationQuery.isLoading && (
        <StatusMessage kind="loading">Calculando correlaciones…</StatusMessage>
      )}

      {!errorMessage && correlation && (
        <>
          <CorrelationHeatmap activos={correlation.activos} matriz={correlation.matriz} />

          <p className="view-note">
            {correlation.fechas_n} fechas comunes usadas ({correlation.method === "pearson" ? "Pearson" : "Spearman"}).
          </p>

          {extremes?.max && extremes?.min && (
            <div className="metric-grid">
              <div className="metric-card">
                <span className="metric-card__label">Par más correlacionado</span>
                <span className="metric-card__value">
                  {extremes.max.a}-{extremes.max.b} ({formatCorr(extremes.max.valor)})
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-card__label">Par menos correlacionado</span>
                <span className="metric-card__value">
                  {extremes.min.a}-{extremes.min.b} ({formatCorr(extremes.min.valor)})
                </span>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
