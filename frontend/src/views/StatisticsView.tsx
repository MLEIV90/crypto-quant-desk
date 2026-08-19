/**
 * Vista "Estadística" (Fase 11) — consume `/api/stats` (Fase 11, sobre
 * `analysis.statistics` + `eda.eda_report.adf_test`, reutilizados tal
 * cual). Cuatro bloques: estacionalidad, estacionariedad, autocorrelación
 * y ciclos — mismo criterio de honestidad que el resto de la app: NADA de
 * esto predice el precio, ver `STATS_INTRO_HELP` y los tooltips de cada
 * bloque en `../helpTexts.ts`.
 *
 * Siempre en diario salvo que se pida explícitamente horario — la
 * estacionalidad horaria (`estacionalidad_horaria`) solo llega si
 * `interval === "1h"` (el backend no la calcula para diario, ver
 * `api/main.py::get_stats`).
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError, getStats } from "../api";
import { BarChart, type BarChartDatum } from "../components/BarChart";
import { InfoTooltip } from "../components/InfoTooltip";
import { MetricCard } from "../components/MetricCard";
import { StatusMessage } from "../components/StatusMessage";
import {
  AUTOCORRELATION_HELP,
  CYCLES_HELP,
  HALVING_HELP,
  SEASONALITY_HELP,
  STATIONARITY_HELP,
  STATS_INTRO_HELP,
} from "../helpTexts";
import type { AutocorrelationPoint, PeriodogramResponse, SeasonalityBucket } from "../types";

const MONTH_LABELS: Record<number, string> = {
  1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
  7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
};

const WEEKDAY_LABELS: Record<number, string> = {
  0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom",
};

function formatPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function seasonalityToBarData(buckets: SeasonalityBucket[], labels?: Record<number, string>): BarChartDatum[] {
  return [...buckets]
    .sort((a, b) => a.bucket - b.bucket)
    .map((b) => {
      const label = labels ? (labels[b.bucket] ?? String(b.bucket)) : String(b.bucket);
      return {
        label,
        value: b.retorno_medio,
        title: `${label}: ${formatPercent(b.retorno_medio)} (n=${b.n})`,
      };
    });
}

function acfToBarData(points: AutocorrelationPoint[], field: "acf_retornos" | "acf_retornos2"): BarChartDatum[] {
  return points.map((p) => {
    const value = p[field] ?? 0;
    return {
      label: p.lag % 5 === 0 ? String(p.lag) : "",
      value,
      title: `lag ${p.lag}: ${value.toFixed(3)}`,
    };
  });
}

function periodogramToBarData(periodograma: PeriodogramResponse): BarChartDatum[] {
  return periodograma.frecuencias.map((freq, index) => {
    const power = periodograma.potencia[index] ?? 0;
    const periodDays = freq > 0 ? 1 / freq : null;
    const showLabel = index % 15 === 0 && periodDays !== null;
    return {
      label: showLabel && periodDays !== null ? `${periodDays.toFixed(0)}d` : "",
      value: power,
      title: periodDays !== null ? `período ~${periodDays.toFixed(1)} días: potencia ${power.toFixed(5)}` : "nivel medio (no es un ciclo)",
    };
  });
}

interface StatisticsViewProps {
  asset: string;
  interval: string;
}

export function StatisticsView({ asset, interval }: StatisticsViewProps) {
  const statsQuery = useQuery({ queryKey: ["stats", asset, interval], queryFn: () => getStats(asset, interval) });
  const stats = statsQuery.data;
  const error = statsQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <p className="view-note">{STATS_INTRO_HELP}</p>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && statsQuery.isLoading && (
        <StatusMessage kind="loading">Calculando estadística de {asset}…</StatusMessage>
      )}

      {!errorMessage && stats && (
        <>
          {/* ESTACIONALIDAD */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Estacionalidad
              <InfoTooltip text={SEASONALITY_HELP.monthly} />
            </h3>
            <div className="stats-grid">
              <div>
                <p className="view-note">Retorno medio por mes</p>
                <BarChart data={seasonalityToBarData(stats.estacionalidad_mensual, MONTH_LABELS)} />
              </div>
              <div>
                <p className="view-note">
                  Retorno medio por día de semana
                  <InfoTooltip text={SEASONALITY_HELP.weekday} placement="bottom" />
                </p>
                <BarChart data={seasonalityToBarData(stats.estacionalidad_semanal, WEEKDAY_LABELS)} />
              </div>
              {stats.estacionalidad_horaria && (
                <div>
                  <p className="view-note">Retorno medio por hora (UTC)</p>
                  <BarChart data={seasonalityToBarData(stats.estacionalidad_horaria)} />
                </div>
              )}
            </div>
          </div>

          {/* ESTACIONARIEDAD */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Estacionariedad (test ADF)
              <InfoTooltip text={STATIONARITY_HELP} />
            </h3>
            <div className="metric-grid">
              <MetricCard
                label="Precio: ¿estacionario?"
                value={stats.adf_precio.es_estacionaria ? "Sí" : "No"}
              />
              <MetricCard label="Precio: p-valor" value={stats.adf_precio.p_valor.toFixed(4)} />
              <MetricCard
                label="Retornos: ¿estacionarios?"
                value={stats.adf_retornos.es_estacionaria ? "Sí" : "No"}
              />
              <MetricCard label="Retornos: p-valor" value={stats.adf_retornos.p_valor.toFixed(4)} />
            </div>
          </div>

          {/* AUTOCORRELACIÓN */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Autocorrelación
              <InfoTooltip text={AUTOCORRELATION_HELP.returns} />
            </h3>
            <div className="stats-grid">
              <div>
                <p className="view-note">ACF de retornos (mercado eficiente ≈ cerca de 0)</p>
                <BarChart data={acfToBarData(stats.autocorrelacion, "acf_retornos")} />
              </div>
              <div>
                <p className="view-note">
                  ACF de retornos² (clustering de volatilidad)
                  <InfoTooltip text={AUTOCORRELATION_HELP.squared} placement="bottom" />
                </p>
                <BarChart data={acfToBarData(stats.autocorrelacion, "acf_retornos2")} />
              </div>
            </div>
          </div>

          {/* CICLOS */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Ciclos (periodograma)
              <InfoTooltip text={CYCLES_HELP} />
            </h3>
            <p className="view-note">Potencia por período (extremo izquierdo = ciclos cortos, derecho = largos)</p>
            <BarChart data={periodogramToBarData(stats.periodograma)} positiveColor="var(--color-accent)" />

            <ul className="stats-periods-list">
              {stats.periodograma.top_periodos.map((p, index) => (
                <li key={index} className="stats-periods-list__row">
                  <span>Período dominante #{index + 1}</span>
                  <strong>{p.periodo_dias.toFixed(1)} días</strong>
                </li>
              ))}
            </ul>

            {stats.halvings_btc && (
              <div className="honesty-banner">
                {HALVING_HELP}
                <ul className="stats-halvings-list">
                  {stats.halvings_btc.map((date) => (
                    <li key={date}>{date}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
