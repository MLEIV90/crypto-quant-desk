/**
 * Vista "Arbitraje" (Fase 12b, completada en Fase 15b) — consume
 * `/api/pairs/screening` y `/api/pairs/detail` (sobre
 * `pairs.cointegration`/`pairs.stability`/`pairs.signals`/`pairs.backtest`,
 * reutilizados tal cual, ver `api/main.py`), más `/api/ohlcv` (Fase 8a,
 * reutilizado) para el scatter.
 *
 * ENCUADRE HONESTO: esto es arbitraje ESTADÍSTICO (pairs trading), NO
 * arbitraje entre exchanges — ver `ARBITRAGE_INTRO_HELP`. La Fase 2 de este
 * proyecto ya mostró que la mayoría de los pares de `config.UNIVERSE` NO
 * están establemente cointegrados; esta vista expone ese resultado tal
 * cual, con su veredicto rojo/verde, en vez de esconderlo. El backtest del
 * par (Fase 15b) lo CONFIRMA cuantitativamente en vez de solo describirlo.
 *
 * El screening (tabla de arriba) siempre es DIARIO — `pairs.stability.screen_pairs_stability`
 * no soporta otro intervalo (ver `api/main.py::get_pairs_screening`) — así
 * que ahí no se usa el `interval` compartido del header. El detalle de un
 * par (spread/z-score/scatter/backtest) sí lo respeta de verdad.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getOhlcv, getPairsDetail, getPairsScreening } from "../api";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type LineSeriesMarkerSpec, type ReferenceLineSpec } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { ScatterChart, type ScatterPoint } from "../components/ScatterChart";
import { StatusMessage } from "../components/StatusMessage";
import {
  ARBITRAGE_CONCEPTS_HELP,
  ARBITRAGE_INTRO_HELP,
  ARBITRAGE_NOT_OPERABLE_WARNING,
  ARBITRAGE_PAIR_BACKTEST_FLAT_PERIODS_NOTE,
  ARBITRAGE_PAIR_BACKTEST_HELP,
  ARBITRAGE_PAIR_BACKTEST_NOT_OPERABLE_WARNING,
  ARBITRAGE_PURPOSE_HEADER,
  ARBITRAGE_SCATTER_HELP,
  ARBITRAGE_SCATTER_NOISE_NOTE,
  ARBITRAGE_SCREENING_HELP,
  ARBITRAGE_ZSCORE_ACTIONABLE_TEXT,
  ARBITRAGE_ZSCORE_EXTREMES_HELP,
  ARBITRAGE_ZSCORE_NOT_ACTIONABLE_TEXT,
} from "../helpTexts";
import { COLORS } from "../theme";

const SCREENING_INTERVAL = "1d";
// Sin límite real de período en /api/pairs/detail (usa TODO el histórico
// disponible) — se pide el mismo volumen para el scatter, así los puntos
// son EXACTAMENTE los que ajustaron la recta de regresión (beta/alpha).
const SCATTER_CANDLE_LIMIT = 60_000;

// Fase 28: umbrales usados para colorear/anotar las métricas del detalle de
// un par — los mismos números que ya describen ARBITRAGE_SCREENING_HELP/
// ARBITRAGE_NOT_OPERABLE_WARNING (60% de ventanas estables) y la
// convención estándar de significancia estadística (p<0.05), más un
// umbral de "vida media útil" (días a pocas semanas) para que half-life
// deje de mostrarse como un número neutro sin interpretación.
const PAIR_ADF_SIGNIFICANCE = 0.05;
const PAIR_STABILITY_THRESHOLD = 0.6;
const PAIR_HALF_LIFE_MAX_USEFUL_DAYS = 30;

/** `half_life_dias` viene en la unidad NATIVA del intervalo (días si es
 * diario, horas si es horario, ver `pairs.cointegration.half_life`) — para
 * comparar contra un umbral pensado en días hace falta convertir primero.
 */
function halfLifeInDays(value: number, interval: string): number {
  return interval === "1h" ? value / 24 : value;
}

const ZSCORE_REFERENCE_LINES: ReferenceLineSpec[] = [
  { price: 2, label: "+2", color: COLORS.danger },
  { price: 0, label: "0", color: COLORS.textMuted },
  { price: -2, label: "-2", color: COLORS.danger },
];

function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(0)}%`;
}

function formatScaledPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)}%`;
}

function formatHalfLife(dias: number | null, interval: string): string {
  if (dias === null) return "Sin dato (no revierte)";
  const unidad = interval === "1h" ? "h" : "d";
  return `${dias.toFixed(1)} ${unidad}`;
}

interface ArbitrageViewProps {
  assets: string[];
  interval: string;
}

export function ArbitrageView({ assets, interval }: ArbitrageViewProps) {
  const [assetY, setAssetY] = useState("ETH");
  const [assetX, setAssetX] = useState("BTC");

  const screeningQuery = useQuery({
    queryKey: ["pairs-screening", SCREENING_INTERVAL],
    queryFn: () => getPairsScreening(SCREENING_INTERVAL),
  });

  const detailQuery = useQuery({
    queryKey: ["pairs-detail", assetY, assetX, interval],
    queryFn: () => getPairsDetail(assetY, assetX, interval),
    enabled: assetY !== assetX,
  });

  // Para el scatter (Fase 15b): mismos activos/intervalo que el detalle,
  // TODO el histórico (ver SCATTER_CANDLE_LIMIT) para que los puntos sean
  // los mismos que ajustaron beta/alpha en el backend.
  const ohlcvYQuery = useQuery({
    queryKey: ["ohlcv", assetY, interval, SCATTER_CANDLE_LIMIT],
    queryFn: () => getOhlcv(assetY, interval, SCATTER_CANDLE_LIMIT),
    enabled: assetY !== assetX,
  });
  const ohlcvXQuery = useQuery({
    queryKey: ["ohlcv", assetX, interval, SCATTER_CANDLE_LIMIT],
    queryFn: () => getOhlcv(assetX, interval, SCATTER_CANDLE_LIMIT),
    enabled: assetY !== assetX,
  });

  const screening = screeningQuery.data;
  const screeningError = screeningQuery.error;
  const screeningErrorMessage =
    screeningError instanceof ApiError ? screeningError.message : screeningError ? String(screeningError) : null;

  const detail = detailQuery.data;
  const detailError = detailQuery.error;
  const detailErrorMessage =
    detailError instanceof ApiError ? detailError.message : detailError ? String(detailError) : null;

  const zscoreMarkers = useMemo((): LineSeriesMarkerSpec[] => {
    if (!detail) return [];
    return detail.zscore_extremos.map((extremo) => ({
      time: extremo.fecha,
      price: extremo.z,
      color: extremo.z > 0 ? COLORS.danger : COLORS.accent,
      shape: extremo.z > 0 ? "arrowUp" : "arrowDown",
    }));
  }, [detail]);

  const zscoreSeries = useMemo(() => {
    if (!detail) return [];
    return [
      {
        id: "zscore",
        label: `z-score(${detail.asset_y}~${detail.asset_x})`,
        color: COLORS.accent,
        data: detail.fechas.map((fecha, i) => ({ time: fecha, value: detail.zscore[i] ?? null })),
        markers: zscoreMarkers,
      },
    ];
  }, [detail, zscoreMarkers]);

  const backtestEquitySeries = useMemo(() => {
    if (!detail) return [];
    return [
      {
        id: "equity",
        label: "Equity (base 1.0)",
        color: COLORS.accent,
        data: detail.backtest.fechas.map((fecha, i) => ({ time: fecha, value: detail.backtest.equity_curve[i] ?? null })),
      },
    ];
  }, [detail]);

  const scatterPoints = useMemo((): ScatterPoint[] => {
    const yData = ohlcvYQuery.data;
    const xData = ohlcvXQuery.data;
    if (!yData || !xData) return [];

    const xByDate = new Map(xData.velas.map((vela) => [vela.fecha, vela.close]));
    const points: ScatterPoint[] = [];
    for (const vela of yData.velas) {
      const xClose = xByDate.get(vela.fecha);
      if (xClose !== undefined && xClose > 0 && vela.close > 0) {
        points.push({ x: Math.log(xClose), y: Math.log(vela.close) });
      }
    }
    return points;
  }, [ohlcvYQuery.data, ohlcvXQuery.data]);

  function handleAssetYChange(next: string) {
    setAssetY(next);
    if (next === assetX) {
      const fallback = assets.find((a) => a !== next);
      if (fallback) setAssetX(fallback);
    }
  }

  function handleAssetXChange(next: string) {
    setAssetX(next);
    if (next === assetY) {
      const fallback = assets.find((a) => a !== next);
      if (fallback) setAssetY(fallback);
    }
  }

  const noEstable = detail ? !(detail.estabilidad?.estable ?? false) : false;

  return (
    <section className="view">
      <div className="honesty-banner">{ARBITRAGE_PURPOSE_HEADER}</div>
      <p className="view-note">{ARBITRAGE_INTRO_HELP}</p>

      <h3 className="panel-subtitle">
        Screening de pares
        <InfoTooltip text={ARBITRAGE_SCREENING_HELP} />
      </h3>

      {screeningErrorMessage && <StatusMessage kind="error">{screeningErrorMessage}</StatusMessage>}
      {!screeningErrorMessage && screeningQuery.isLoading && (
        <StatusMessage kind="loading">Corriendo cointegración rolling sobre todos los pares…</StatusMessage>
      )}

      {screening && (
        <>
          <p className="view-note">
            {screening.n_estables} de {screening.n_total} pares operables (fracción cointegrada rolling ≥ 60%).
          </p>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Par</th>
                <th>Dirección</th>
                <th>% ventanas cointegradas</th>
                <th>Beta medio</th>
                <th>Operable</th>
              </tr>
            </thead>
            <tbody>
              {screening.filas.map((fila) => (
                <tr key={fila.par}>
                  <td>{fila.par}</td>
                  <td>{fila.direccion}</td>
                  <td>
                    <div className="pair-threshold-cell">
                      <span className="pair-threshold-cell__label">
                        <span style={{ color: fila.estable ? COLORS.success : COLORS.danger }}>
                          {formatPercent(fila.fraccion_cointegrada)}
                        </span>
                        <span className="pair-threshold-cell__target">necesita 60%</span>
                      </span>
                      <div className="pair-threshold-cell__track">
                        <div
                          className="pair-threshold-cell__fill"
                          style={{
                            width: `${Math.min(100, fila.fraccion_cointegrada * 100)}%`,
                            background: fila.estable ? COLORS.success : COLORS.danger,
                          }}
                        />
                        <div className="pair-threshold-cell__threshold-marker" style={{ left: "60%" }} />
                      </div>
                    </div>
                  </td>
                  <td>
                    {fila.beta_medio.toFixed(3)} ± {fila.beta_std.toFixed(3)}
                  </td>
                  <td>
                    <span className={`pair-semaphore pair-semaphore--${fila.estable ? "ok" : "no"}`}>
                      {fila.estable ? "Sí" : "No"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h3 className="panel-subtitle">Detalle de un par</h3>

      <div className="asset-selector">
        <label className="asset-selector__field">
          Activo Y (dependiente)
          <select value={assetY} onChange={(event) => handleAssetYChange(event.target.value)}>
            {assets.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label className="asset-selector__field">
          Activo X
          <select value={assetX} onChange={(event) => handleAssetXChange(event.target.value)}>
            {assets.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>

      {detailErrorMessage && <StatusMessage kind="error">{detailErrorMessage}</StatusMessage>}
      {!detailErrorMessage && detailQuery.isLoading && (
        <StatusMessage kind="loading">
          Analizando {assetY}~{assetX}…
        </StatusMessage>
      )}

      {detail && (
        <>
          {noEstable && <div className="honesty-banner">{ARBITRAGE_NOT_OPERABLE_WARNING}</div>}

          <div className="metric-grid">
            <MetricCard
              label="¿Cointegrado? (in-sample)"
              value={detail.es_cointegrado ? "Sí" : "No"}
              valueColor={detail.es_cointegrado ? COLORS.success : COLORS.danger}
              help={ARBITRAGE_CONCEPTS_HELP.cointegracion}
            />
            <MetricCard
              label="p-valor ADF"
              value={detail.p_valor_adf.toFixed(4)}
              valueColor={detail.p_valor_adf < PAIR_ADF_SIGNIFICANCE ? COLORS.success : COLORS.danger}
              help={ARBITRAGE_CONCEPTS_HELP.cointegracion}
              subtext={
                detail.p_valor_adf < PAIR_ADF_SIGNIFICANCE
                  ? "< 0.05: indica cointegración (in-sample)"
                  : "necesita < 0.05 para indicar cointegración"
              }
              subtextColor={detail.p_valor_adf < PAIR_ADF_SIGNIFICANCE ? undefined : COLORS.danger}
            />
            <MetricCard
              label="Half-life"
              value={formatHalfLife(detail.half_life_dias, detail.interval)}
              help={ARBITRAGE_CONCEPTS_HELP.halfLife}
              valueColor={
                detail.half_life_dias === null || halfLifeInDays(detail.half_life_dias, detail.interval) > PAIR_HALF_LIFE_MAX_USEFUL_DAYS
                  ? COLORS.danger
                  : COLORS.success
              }
              subtext={
                detail.half_life_dias === null
                  ? "no revierte — sin vida media válida"
                  : halfLifeInDays(detail.half_life_dias, detail.interval) > PAIR_HALF_LIFE_MAX_USEFUL_DAYS
                    ? "en la práctica no revierte de forma operable (útil: días a pocas semanas)"
                    : "dentro del rango útil (días a pocas semanas)"
              }
              subtextColor={
                detail.half_life_dias === null || halfLifeInDays(detail.half_life_dias, detail.interval) > PAIR_HALF_LIFE_MAX_USEFUL_DAYS
                  ? COLORS.danger
                  : undefined
              }
            />
            <MetricCard
              label="% ventanas estables"
              value={detail.estabilidad ? formatPercent(detail.estabilidad.fraccion_cointegrada) : "Sin dato"}
              help={ARBITRAGE_CONCEPTS_HELP.estabilidad}
              valueColor={
                detail.estabilidad
                  ? detail.estabilidad.fraccion_cointegrada >= PAIR_STABILITY_THRESHOLD
                    ? COLORS.success
                    : COLORS.danger
                  : undefined
              }
              subtext={
                detail.estabilidad
                  ? detail.estabilidad.fraccion_cointegrada >= PAIR_STABILITY_THRESHOLD
                    ? "≥ 60%: cumple el umbral de operable"
                    : "necesita ≥ 60% para considerarse operable"
                  : undefined
              }
              subtextColor={
                detail.estabilidad && detail.estabilidad.fraccion_cointegrada < PAIR_STABILITY_THRESHOLD
                  ? COLORS.danger
                  : undefined
              }
            />
            <MetricCard
              label="Veredicto"
              value={noEstable ? "NO operable" : "Operable"}
              valueColor={noEstable ? COLORS.danger : COLORS.success}
            />
          </div>

          <div className="zscore-indicator">
            <span className="zscore-indicator__label">
              Z-score actual del spread
              <InfoTooltip text={ARBITRAGE_CONCEPTS_HELP.zscore} placement="bottom" />
            </span>
            <span
              className="zscore-indicator__value"
              style={{
                color:
                  detail.zscore_actual !== null && Math.abs(detail.zscore_actual) > 2 ? COLORS.danger : COLORS.text,
              }}
            >
              {detail.zscore_actual !== null ? detail.zscore_actual.toFixed(2) : "—"}
            </span>
            <span className="zscore-indicator__interpretation">{detail.zscore_interpretacion}</span>
          </div>
          <p className="view-note">{noEstable ? ARBITRAGE_ZSCORE_NOT_ACTIONABLE_TEXT : ARBITRAGE_ZSCORE_ACTIONABLE_TEXT}</p>

          <h3 className="panel-subtitle">
            Spread (z-score) con bandas ±2 / 0
            <InfoTooltip text={ARBITRAGE_CONCEPTS_HELP.spread} />
          </h3>
          <p className="view-note">
            Las flechas marcan los extremos históricos (|z| ≥ 2): rojas hacia arriba (spread muy estirado hacia
            arriba), celestes hacia abajo (muy estirado hacia abajo).
            <InfoTooltip text={ARBITRAGE_ZSCORE_EXTREMES_HELP} placement="bottom" />
          </p>
          <LineChartPanel series={zscoreSeries} height={320} referenceLines={ZSCORE_REFERENCE_LINES} />

          {detail.estabilidad_mensaje && (
            <p className="view-note">{detail.estabilidad_mensaje}</p>
          )}

          <h3 className="panel-subtitle">
            Dispersión y recta de regresión
            <InfoTooltip text={ARBITRAGE_SCATTER_HELP} />
          </h3>
          <p className="view-note">{ARBITRAGE_SCATTER_NOISE_NOTE}</p>
          {ohlcvYQuery.isLoading || ohlcvXQuery.isLoading ? (
            <StatusMessage kind="loading">Cargando precios para el scatter…</StatusMessage>
          ) : (
            <ScatterChart
              points={scatterPoints}
              alpha={detail.alpha}
              beta={detail.beta}
              xLabel={`log(precio ${detail.asset_x})`}
              yLabel={`log(precio ${detail.asset_y})`}
              height={340}
            />
          )}

          <h3 className="panel-subtitle">
            Backtest del par
            <InfoTooltip text={ARBITRAGE_PAIR_BACKTEST_HELP} />
          </h3>
          {noEstable && <div className="honesty-banner">{ARBITRAGE_PAIR_BACKTEST_NOT_OPERABLE_WARNING}</div>}
          <div className="metric-grid">
            <MetricCard
              label="Retorno total"
              value={formatScaledPercent(detail.backtest.metrics.total_return)}
              valueColor={detail.backtest.metrics.total_return >= 0 ? COLORS.success : COLORS.danger}
            />
            <MetricCard
              label="Sharpe"
              value={detail.backtest.metrics.sharpe.toFixed(2)}
              valueColor={detail.backtest.metrics.sharpe >= 0 ? COLORS.success : COLORS.danger}
            />
            <MetricCard
              label="Máximo drawdown"
              value={formatScaledPercent(detail.backtest.metrics.max_drawdown)}
              valueColor={COLORS.danger}
            />
            <MetricCard label="Cantidad de operaciones" value={detail.backtest.metrics.n_trades} />
          </div>
          <LineChartPanel series={backtestEquitySeries} height={280} />
          <p className="view-note">{ARBITRAGE_PAIR_BACKTEST_FLAT_PERIODS_NOTE}</p>
        </>
      )}
    </section>
  );
}
