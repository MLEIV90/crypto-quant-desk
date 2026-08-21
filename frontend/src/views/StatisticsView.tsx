/**
 * Vista "Ciclos y Estadística" (Fase 11, REHECHA en Fase 15a) — consume
 * `/api/stats` (`analysis.statistics` + `analysis.cycles` + `eda.eda_report.adf_test`,
 * reutilizados tal cual). Mismo criterio de honestidad de siempre: NADA de
 * esto predice el precio, ver `STATS_INTRO_HELP` y los tooltips de cada
 * bloque en `../helpTexts.ts`.
 *
 * Fase 15a: se sacó el periodograma de ciclos de 2-3 días (ruido de alta
 * frecuencia, sin significado de mercado) y también el bar chart de
 * estacionalidad MENSUAL agregada (un solo promedio por mes, mezclando
 * todos los años) — lo reemplaza el mapa de calor mes x año de acá abajo,
 * que muestra la estacionalidad real año por año en vez de un promedio que
 * esconde qué tan distinto fue cada año. Se agregan drawdowns históricos,
 * fases de mercado bull/bear, y (solo BTC) ciclos de halving.
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
import { MonthlyHeatmap } from "../components/MonthlyHeatmap";
import { StatusMessage } from "../components/StatusMessage";
import {
  AUTOCORRELATION_HELP,
  DRAWDOWN_HELP,
  HALVING_CYCLE_HELP,
  MARKET_PHASES_HELP,
  MONTHLY_HEATMAP_HELP,
  SEASONALITY_HELP,
  STATIONARITY_HELP,
  STATS_INTRO_HELP,
} from "../helpTexts";
import { COLORS } from "../theme";
import type { AutocorrelationPoint, SeasonalityBucket } from "../types";

const WEEKDAY_LABELS: Record<number, string> = {
  0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom",
};

function formatDecimalPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

/** Los campos de ciclos (drawdowns/fases/halving) ya vienen en PUNTOS
 * PORCENTUALES desde el backend (p. ej. -50.0 == -50%), a diferencia de
 * `estacionalidad_*`/`autocorrelacion` (escala decimal, 0.01 == 1%) — dos
 * formatters distintos a propósito, para no mezclar las dos escalas.
 */
function formatScaledPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString();
}

function seasonalityToBarData(buckets: SeasonalityBucket[], labels?: Record<number, string>): BarChartDatum[] {
  return [...buckets]
    .sort((a, b) => a.bucket - b.bucket)
    .map((b) => {
      const label = labels ? (labels[b.bucket] ?? String(b.bucket)) : String(b.bucket);
      return {
        label,
        value: b.retorno_medio,
        title: `${label}: ${formatDecimalPercent(b.retorno_medio)} (n=${b.n})`,
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
        <StatusMessage kind="loading">Calculando ciclos y estadística de {asset}…</StatusMessage>
      )}

      {!errorMessage && stats && (
        <>
          {/* ESTACIONALIDAD (día de semana / hora) */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Estacionalidad
              <InfoTooltip text={SEASONALITY_HELP.weekday} />
            </h3>
            <div className="stats-grid">
              <div>
                <p className="view-note">Retorno medio por día de semana</p>
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
              <MetricCard label="Precio: ¿estacionario?" value={stats.adf_precio.es_estacionaria ? "Sí" : "No"} />
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

          {/* DRAWDOWNS HISTÓRICOS */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Drawdowns históricos
              <InfoTooltip text={DRAWDOWN_HELP} />
            </h3>
            {stats.drawdowns.length === 0 ? (
              <p className="view-note">Sin drawdowns registrados en el período disponible.</p>
            ) : (
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Pico</th>
                    <th>Fondo</th>
                    <th>Profundidad</th>
                    <th>Días de caída</th>
                    <th>Recuperación</th>
                    <th>Días de recuperación</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.drawdowns.map((d, index) => (
                    <tr key={index}>
                      <td>{formatDate(d.fecha_pico)}</td>
                      <td>{formatDate(d.fecha_fondo)}</td>
                      <td style={{ color: COLORS.danger }}>{formatScaledPercent(d.profundidad_pct)}</td>
                      <td>{d.dias_caida}</td>
                      <td>{d.fecha_recuperacion ? formatDate(d.fecha_recuperacion) : "Todavía no recuperó"}</td>
                      <td>{d.dias_recuperacion ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* FASES DE MERCADO */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Fases de mercado
              <InfoTooltip text={MARKET_PHASES_HELP} />
            </h3>
            {stats.fases_mercado.length === 0 ? (
              <p className="view-note">Ningún movimiento cruzó el umbral de 20% en el período disponible.</p>
            ) : (
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Tipo</th>
                    <th>Inicio</th>
                    <th>Fin</th>
                    <th>Duración</th>
                    <th>Retorno</th>
                    <th>Estado</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.fases_mercado.map((f, index) => (
                    <tr key={index}>
                      <td style={{ color: f.tipo === "bull" ? COLORS.success : COLORS.danger, fontWeight: 700 }}>
                        {f.tipo === "bull" ? "Alcista" : "Bajista"}
                      </td>
                      <td>{formatDate(f.fecha_inicio)}</td>
                      <td>{formatDate(f.fecha_fin)}</td>
                      <td>{f.duracion_dias} días</td>
                      <td style={{ color: f.retorno_pct >= 0 ? COLORS.success : COLORS.danger }}>
                        {formatScaledPercent(f.retorno_pct)}
                      </td>
                      <td>{f.confirmada ? "Confirmada" : "En curso"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {asset === "BTC" && stats.ciclos_halving && (
              <>
                <h4 className="stats-section__title">
                  Ciclos de halving
                  <InfoTooltip text={HALVING_CYCLE_HELP} placement="bottom" />
                </h4>
                <div className="honesty-banner">
                  Solo {stats.ciclos_halving.n_halvings_con_datos} de los {stats.ciclos_halving.n_halvings_totales}{" "}
                  halvings de Bitcoin caen dentro del histórico de precios disponible — n={stats.ciclos_halving.n_halvings_con_datos}{" "}
                  es una muestra estadísticamente insuficiente para tratar el "ciclo de 4 años" como una regla.
                </div>
                {stats.ciclos_halving.ciclos.length > 0 && (
                  <table className="metrics-table">
                    <thead>
                      <tr>
                        <th>Inicio</th>
                        <th>Fin</th>
                        <th>Estado</th>
                        <th>Duración</th>
                        <th>Retorno</th>
                        <th>Drawdown máximo</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.ciclos_halving.ciclos.map((c, index) => (
                        <tr key={index}>
                          <td>{formatDate(c.fecha_inicio)}</td>
                          <td>{formatDate(c.fecha_fin)}</td>
                          <td>{c.en_curso ? "En curso" : "Completo"}</td>
                          <td>{c.duracion_dias} días</td>
                          <td style={{ color: c.retorno_pct >= 0 ? COLORS.success : COLORS.danger }}>
                            {formatScaledPercent(c.retorno_pct)}
                          </td>
                          <td style={{ color: COLORS.danger }}>{formatScaledPercent(c.drawdown_maximo_pct)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </>
            )}
          </div>

          {/* MAPA DE CALOR MES x AÑO */}
          <div className="stats-section">
            <h3 className="stats-section__title">
              Retorno por mes y año
              <InfoTooltip text={MONTHLY_HEATMAP_HELP} />
            </h3>
            {stats.heatmap_mensual.anios.length === 0 ? (
              <p className="view-note">Sin suficiente historia para armar el mapa de calor.</p>
            ) : (
              <div className="stats-heatmap-wrapper">
                <MonthlyHeatmap anios={stats.heatmap_mensual.anios} matriz={stats.heatmap_mensual.matriz} />
              </div>
            )}
          </div>
        </>
      )}
    </section>
  );
}
