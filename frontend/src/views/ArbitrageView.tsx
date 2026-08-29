/**
 * Vista "Arbitraje" (Fase 12b, completada en Fase 15b, rehecha como PANEL
 * DE PARES en Fase 30) — consume `/api/pairs/screening` y
 * `/api/pairs/detail` (sobre `pairs.cointegration`/`pairs.kalman_hedge`/
 * `pairs.stability`/`pairs.signals`/`pairs.backtest`, reutilizados tal
 * cual, ver `api/main.py`), `/api/correlation` (Fase 13b, reutilizado para
 * la correlación actual del par) y `/api/ohlcv` (Fase 8a, reutilizado)
 * para el scatter.
 *
 * ENCUADRE HONESTO: esto es arbitraje ESTADÍSTICO (pairs trading), NO
 * arbitraje entre exchanges — ver `ARBITRAGE_INTRO_HELP`. La Fase 2 de este
 * proyecto ya mostró que la mayoría de los pares de `config.UNIVERSE` NO
 * están establemente cointegrados; esta vista expone ese resultado tal
 * cual, con los NÚMEROS CRUDOS y el CRITERIO EN MANOS DEL USUARIO (Fase
 * 30): nada de umbrales fijos decididos por defecto sin poder cambiarlos —
 * los umbrales de entrada/salida/stop del backtest, el umbral de
 * estabilidad y el ancho de las bandas del spread son todos configurables
 * acá mismo, y el veredicto se recalcula con el criterio que el usuario
 * elija, no con uno escondido en el backend.
 *
 * El screening (tabla de arriba) siempre es DIARIO — `pairs.stability.screen_pairs_stability`
 * no soporta otro intervalo (ver `api/main.py::get_pairs_screening`) — así
 * que ahí no se usa el `interval` compartido del header, y su umbral de
 * operable queda fijo en 60% (es un ranking de referencia rápida, no el
 * panel de un par). El panel de un par SÍ respeta `interval` y todos los
 * criterios configurables de Fase 30.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getCorrelation, getOhlcv, getPairsDetail, getPairsScreening } from "../api";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type LineSeriesMarkerSpec, type ReferenceLineSpec } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { candleLimitForPeriod, DEFAULT_PERIOD, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { ScatterChart, type ScatterPoint } from "../components/ScatterChart";
import { StatusMessage } from "../components/StatusMessage";
import {
  arbitrageNotOperableWarning,
  arbitrageZscoreActionableText,
  arbitrageZscoreNotActionableText,
  ARBITRAGE_BANDS_HELP,
  ARBITRAGE_CONCEPTS_HELP,
  ARBITRAGE_INTRO_HELP,
  ARBITRAGE_KALMAN_HELP,
  ARBITRAGE_LONG_ONLY_HELP,
  ARBITRAGE_PAIR_BACKTEST_FLAT_PERIODS_NOTE,
  ARBITRAGE_PAIR_BACKTEST_HELP,
  ARBITRAGE_PAIR_BACKTEST_NOT_OPERABLE_WARNING,
  ARBITRAGE_PURPOSE_HEADER,
  ARBITRAGE_RATIO_HELP,
  ARBITRAGE_SCATTER_HELP,
  ARBITRAGE_SCATTER_NOISE_NOTE,
  ARBITRAGE_SCREENING_HELP,
  ARBITRAGE_STABILITY_THRESHOLD_HELP,
  ARBITRAGE_STABILITY_THRESHOLD_ONE_LINER,
  ARBITRAGE_ZSCORE_EXTREMES_HELP,
  ARBITRAGE_ZSCORE_ZONES_HELP,
} from "../helpTexts";
import { COLORS } from "../theme";

const SCREENING_INTERVAL = "1d";
// Sin límite real de período en /api/pairs/detail (usa TODO el histórico
// disponible) — se pide el mismo volumen para el scatter y la correlación,
// así los puntos son EXACTAMENTE los que ajustaron la recta de regresión
// (beta/alpha) y el par de fechas del backtest.
const SCATTER_CANDLE_LIMIT = 60_000;

// Fase 28: umbrales usados para colorear/anotar p-valor ADF y half-life —
// convención estándar de significancia (p<0.05) y una noción de "vida
// media útil" (días a pocas semanas). Fase 30: el umbral de ESTABILIDAD ya
// NO es una constante acá — es `stabilityThreshold`, elegido por el
// usuario (ver el slider del Componente 5).
const PAIR_ADF_SIGNIFICANCE = 0.05;
const PAIR_HALF_LIFE_MAX_USEFUL_DAYS = 30;

// Fase 30: defaults de los criterios configurables — los mismos que ya
// traía el backend por defecto (pairs.backtest.DEFAULT_PAIR_*,
// pairs.stability.DEFAULT_STABLE_FRACTION_THRESHOLD, config.TRANSACTION_COST_BPS),
// para que el panel arranque mostrando exactamente lo que se pediría sin
// pasar ningún parámetro.
const DEFAULT_BT_ENTRY = 2.0;
const DEFAULT_BT_EXIT = 0.5;
const DEFAULT_BT_STOP = 3.0;
const DEFAULT_BT_COST_BPS = 10;
const DEFAULT_STABILITY_THRESHOLD = 0.6;
const DEFAULT_BAND_N_STD = 2.0;

type RatioMode = "simple" | "log_spread";

/** `half_life_dias` viene en la unidad NATIVA del intervalo (días si es
 * diario, horas si es horario, ver `pairs.cointegration.half_life`) — para
 * comparar contra un umbral pensado en días hace falta convertir primero.
 */
function halfLifeInDays(value: number, interval: string): number {
  return interval === "1h" ? value / 24 : value;
}

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

function formatCorr(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

/** Recorta un array de series alineadas a `fechas` a los últimos `limit`
 * puntos — Fase 30: el período elegido (Componente 1) solo afecta QUÉ
 * TRAMO de las series se GRAFICA, no el cálculo en sí (beta, cointegración,
 * Kalman y el backtest siguen usando TODO el histórico disponible, igual
 * que antes de este componente).
 */
function trimTail<T>(values: T[], limit: number): T[] {
  if (values.length <= limit) return values;
  return values.slice(values.length - limit);
}

interface ArbitrageViewProps {
  assets: string[];
  interval: string;
}

export function ArbitrageView({ assets, interval }: ArbitrageViewProps) {
  const [assetY, setAssetY] = useState("ETH");
  const [assetX, setAssetX] = useState("BTC");

  // Fase 30: período de VISUALIZACIÓN para los gráficos de ratio/spread/
  // bandas/Kalman/z-score (Componentes 1-4) — reutiliza PeriodSelector tal
  // cual, mismo patrón que Estadística/Comparación/Correlación.
  const [period, setPeriod] = useState<PeriodKey>(DEFAULT_PERIOD);
  const [ratioMode, setRatioMode] = useState<RatioMode>("simple");
  const [bandNStd, setBandNStd] = useState(DEFAULT_BAND_N_STD);
  const [stabilityThreshold, setStabilityThreshold] = useState(DEFAULT_STABILITY_THRESHOLD);
  const [btEntry, setBtEntry] = useState(DEFAULT_BT_ENTRY);
  const [btExit, setBtExit] = useState(DEFAULT_BT_EXIT);
  const [btStop, setBtStop] = useState(DEFAULT_BT_STOP);
  const [btCostBps, setBtCostBps] = useState(DEFAULT_BT_COST_BPS);
  const [btLongOnly, setBtLongOnly] = useState(false);

  const screeningQuery = useQuery({
    queryKey: ["pairs-screening", SCREENING_INTERVAL],
    queryFn: () => getPairsScreening(SCREENING_INTERVAL),
  });

  const detailQuery = useQuery({
    queryKey: [
      "pairs-detail",
      assetY,
      assetX,
      interval,
      btEntry,
      btExit,
      btStop,
      btCostBps,
      btLongOnly,
      stabilityThreshold,
      bandNStd,
    ],
    queryFn: () =>
      getPairsDetail(assetY, assetX, interval, {
        btEntry,
        btExit,
        btStop,
        btCostBps,
        btLongOnly,
        stabilityThreshold,
        bandNStd,
      }),
    enabled: assetY !== assetX,
  });

  // Fase 30 (Componente 5): correlación ACTUAL del par — reutiliza
  // `/api/correlation` (Fase 13b) tal cual en vez de recalcularla acá, así
  // el número coincide siempre con el que muestra la vista "Correlación".
  const correlationQuery = useQuery({
    queryKey: ["correlation-for-pair", interval, SCATTER_CANDLE_LIMIT],
    queryFn: () => getCorrelation(interval, SCATTER_CANDLE_LIMIT, "pearson"),
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

  const correlationActual = useMemo(() => {
    const correlation = correlationQuery.data;
    if (!correlation) return null;
    const iY = correlation.activos.indexOf(assetY);
    const iX = correlation.activos.indexOf(assetX);
    if (iY === -1 || iX === -1) return null;
    return correlation.matriz[iY][iX];
  }, [correlationQuery.data, assetY, assetX]);

  // Fase 30: recorta TODAS las series alineadas a `fechas` al período
  // elegido, de una sola vez — el cálculo (full histórico) no cambia, solo
  // lo que se grafica.
  const trimmed = useMemo(() => {
    if (!detail) return null;
    const limit = candleLimitForPeriod(period, interval);
    const n = Math.min(limit, detail.fechas.length);
    const fechas = trimTail(detail.fechas, limit);
    const startFecha = fechas[0];
    return {
      fechas,
      ratio: trimTail(detail.ratio, limit),
      spread: trimTail(detail.spread, limit),
      bandaMedia: trimTail(detail.banda_media, limit),
      bandaSuperior: trimTail(detail.banda_superior, limit),
      bandaInferior: trimTail(detail.banda_inferior, limit),
      kalmanBeta: trimTail(detail.kalman_beta, limit),
      zscore: trimTail(detail.zscore, limit),
      n,
      startFecha,
    };
  }, [detail, period, interval]);

  const zscoreMarkers = useMemo((): LineSeriesMarkerSpec[] => {
    if (!detail || !trimmed) return [];
    return detail.zscore_extremos
      .filter((extremo) => !trimmed.startFecha || extremo.fecha >= trimmed.startFecha)
      .map((extremo) => ({
        time: extremo.fecha,
        price: extremo.z,
        color: extremo.z > 0 ? COLORS.danger : COLORS.accent,
        shape: extremo.z > 0 ? "arrowUp" : "arrowDown",
      }));
  }, [detail, trimmed]);

  const zscoreSeries = useMemo(() => {
    if (!detail || !trimmed) return [];
    return [
      {
        id: "zscore",
        label: `z-score(${detail.asset_y}~${detail.asset_x})`,
        color: COLORS.accent,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: trimmed.zscore[i] ?? null })),
        markers: zscoreMarkers,
      },
    ];
  }, [detail, trimmed, zscoreMarkers]);

  const zscoreReferenceLines = useMemo((): ReferenceLineSpec[] => {
    return [
      { price: btEntry, label: `+entrada (${btEntry})`, color: COLORS.danger },
      { price: btExit, label: `+salida (${btExit})`, color: COLORS.textMuted },
      { price: 0, label: "0", color: COLORS.textMuted },
      { price: -btExit, label: `-salida (${btExit})`, color: COLORS.textMuted },
      { price: -btEntry, label: `-entrada (${btEntry})`, color: COLORS.danger },
    ];
  }, [btEntry, btExit]);

  // Fase 30, Componente 1: ratio crudo o spread log, según el toggle.
  const ratioSeries = useMemo(() => {
    if (!detail || !trimmed) return [];
    const isSimple = ratioMode === "simple";
    const values = isSimple ? trimmed.ratio : trimmed.spread;
    return [
      {
        id: "ratio",
        label: isSimple
          ? `Ratio ${detail.asset_y}/${detail.asset_x}`
          : `Spread log(${detail.asset_y}) - β·log(${detail.asset_x})`,
        color: COLORS.accent,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: values[i] ?? null })),
      },
    ];
  }, [detail, trimmed, ratioMode]);

  // Fase 30, Componente 2: spread + bandas, con marcas donde el spread
  // salió de la banda (zona "extrema" respecto de su comportamiento reciente).
  const bandExceedMarkers = useMemo((): LineSeriesMarkerSpec[] => {
    if (!trimmed) return [];
    const markers: LineSeriesMarkerSpec[] = [];
    for (let i = 0; i < trimmed.fechas.length; i++) {
      const s = trimmed.spread[i];
      const up = trimmed.bandaSuperior[i];
      const lo = trimmed.bandaInferior[i];
      if (s === null || s === undefined) continue;
      if (up !== null && up !== undefined && s > up) {
        markers.push({ time: trimmed.fechas[i], price: s, color: COLORS.danger, shape: "arrowUp" });
      } else if (lo !== null && lo !== undefined && s < lo) {
        markers.push({ time: trimmed.fechas[i], price: s, color: COLORS.accent, shape: "arrowDown" });
      }
    }
    return markers;
  }, [trimmed]);

  const bandsSeries = useMemo(() => {
    if (!detail || !trimmed) return [];
    return [
      {
        id: "spread",
        label: "Spread",
        color: COLORS.accent,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: trimmed.spread[i] ?? null })),
        markers: bandExceedMarkers,
      },
      {
        id: "media",
        label: "Media móvil",
        color: COLORS.textMuted,
        lineWidth: 1 as const,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: trimmed.bandaMedia[i] ?? null })),
      },
      {
        id: "superior",
        label: `+${bandNStd}σ`,
        color: COLORS.danger,
        lineWidth: 1 as const,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: trimmed.bandaSuperior[i] ?? null })),
      },
      {
        id: "inferior",
        label: `-${bandNStd}σ`,
        color: COLORS.danger,
        lineWidth: 1 as const,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: trimmed.bandaInferior[i] ?? null })),
      },
    ];
  }, [detail, trimmed, bandExceedMarkers, bandNStd]);

  // Fase 30, Componente 3: hedge ratio DINÁMICO (Kalman), comparado contra
  // el único beta estático (OLS) de toda la muestra vía una línea de referencia.
  const kalmanSeries = useMemo(() => {
    if (!detail || !trimmed) return [];
    return [
      {
        id: "kalman-beta",
        label: `Beta Kalman (${detail.asset_y}~${detail.asset_x})`,
        color: COLORS.accent,
        data: trimmed.fechas.map((fecha, i) => ({ time: fecha, value: trimmed.kalmanBeta[i] ?? null })),
      },
    ];
  }, [detail, trimmed]);

  const kalmanReferenceLines = useMemo((): ReferenceLineSpec[] => {
    if (!detail) return [];
    return [{ price: detail.beta, label: "beta OLS (estático)", color: COLORS.textMuted }];
  }, [detail]);

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
  const stabilityThresholdPct = Math.round(stabilityThreshold * 100);

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
            {screening.n_estables} de {screening.n_total} pares operables (fracción cointegrada rolling ≥ 60%,
            umbral fijo de este ranking de referencia — el panel de abajo deja elegir el propio).
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

      <h3 className="panel-subtitle">Panel del par</h3>

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

      {detail && trimmed && (
        <>
          {noEstable && <div className="honesty-banner">{arbitrageNotOperableWarning(stabilityThresholdPct)}</div>}

          <h3 className="panel-subtitle">
            1 — Ratio / spread en el tiempo
            <InfoTooltip text={ARBITRAGE_RATIO_HELP} />
          </h3>
          <p className="view-note">
            Cómo cambió la relación entre {detail.asset_y} y {detail.asset_x} con el tiempo — no un único número,
            sino la película completa.
          </p>
          <div className="chart-controls-row">
            <PeriodSelector active={period} onChange={setPeriod} />
            <div className="period-selector" role="group" aria-label="Modo del ratio">
              <button
                type="button"
                className={`period-selector__pill${ratioMode === "simple" ? " period-selector__pill--active" : ""}`}
                onClick={() => setRatioMode("simple")}
              >
                Ratio simple
              </button>
              <button
                type="button"
                className={`period-selector__pill${ratioMode === "log_spread" ? " period-selector__pill--active" : ""}`}
                onClick={() => setRatioMode("log_spread")}
              >
                Spread log
              </button>
            </div>
          </div>
          <LineChartPanel series={ratioSeries} height={280} />

          <h3 className="panel-subtitle">
            2 — Spread con bandas (media móvil ± {bandNStd}σ)
            <InfoTooltip text={ARBITRAGE_BANDS_HELP} />
          </h3>
          <p className="view-note">
            Las flechas marcan los momentos donde el spread salió de la banda: rojas hacia arriba (por encima de
            +{bandNStd}σ), celestes hacia abajo (por debajo de -{bandNStd}σ).
          </p>
          <div className="pair-controls">
            <label className="pair-controls__field">
              Ancho de banda (N desvíos)
              <span className="pair-controls__field-value">{bandNStd.toFixed(1)}σ</span>
              <input
                type="range"
                className="pair-controls__slider"
                min={0.5}
                max={4}
                step={0.5}
                value={bandNStd}
                onChange={(event) => setBandNStd(Number(event.target.value))}
              />
            </label>
          </div>
          <LineChartPanel series={bandsSeries} height={320} />

          <h3 className="panel-subtitle">
            3 — Relación dinámica: hedge ratio (Kalman) en el tiempo
            <InfoTooltip text={ARBITRAGE_KALMAN_HELP} />
          </h3>
          <p className="view-note">
            La línea gris punteada es el ÚNICO beta estático (OLS, toda la muestra a la vez, {detail.beta.toFixed(3)}
            ) — la línea de color es cómo ese "tipo de cambio de equilibrio" fue cambiando día a día.
          </p>
          <LineChartPanel series={kalmanSeries} height={260} referenceLines={kalmanReferenceLines} />

          <h3 className="panel-subtitle">
            4 — Z-score con zonas de entrada/salida
            <InfoTooltip text={ARBITRAGE_ZSCORE_ZONES_HELP} />
          </h3>
          <p className="view-note">
            Las flechas marcan los extremos históricos (|z| ≥ 2). Los umbrales de las líneas se configuran en el
            backtest de más abajo (Componente 6) — son los mismos.
            <InfoTooltip text={ARBITRAGE_ZSCORE_EXTREMES_HELP} placement="bottom" />
          </p>
          <LineChartPanel series={zscoreSeries} height={320} referenceLines={zscoreReferenceLines} />

          <div className="zscore-indicator">
            <span className="zscore-indicator__label">
              Z-score actual del spread
              <InfoTooltip text={ARBITRAGE_CONCEPTS_HELP.zscore} placement="bottom" />
            </span>
            <span
              className="zscore-indicator__value"
              style={{
                color:
                  detail.zscore_actual !== null && Math.abs(detail.zscore_actual) > btEntry
                    ? COLORS.danger
                    : COLORS.text,
              }}
            >
              {detail.zscore_actual !== null ? detail.zscore_actual.toFixed(2) : "—"}
            </span>
            <span className="zscore-indicator__interpretation">{detail.zscore_interpretacion}</span>
          </div>
          <p className="view-note">
            {noEstable
              ? arbitrageZscoreNotActionableText(stabilityThresholdPct)
              : arbitrageZscoreActionableText(stabilityThresholdPct)}
          </p>

          <h3 className="panel-subtitle">5 — Métricas de la relación</h3>
          <div className="metric-grid">
            <MetricCard
              label="Correlación (actual)"
              value={correlationActual !== null ? formatCorr(correlationActual) : "Cargando…"}
              help="Correlación de Pearson entre los retornos de las dos monedas — ver la vista Correlación para el detalle. Alta correlación NO implica cointegración: dos activos pueden moverse muy parecido sin que exista un spread estable entre ellos."
            />
            <MetricCard
              label="p-valor ADF (in-sample)"
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
          </div>

          {/* Fase 32 (hallazgo A3-02): el umbral de estabilidad, la fracción
              cointegrada de ESTE par y el veredicto viven todos JUNTOS acá —
              antes el slider estaba suelto varios elementos más abajo, lejos
              de la métrica que en realidad controla, y no había ningún texto
              que dijera "por qué" el veredicto cambia o no al moverlo. */}
          <div className="stability-panel">
            <div className="stability-panel__header">
              <h4 className="stability-panel__title">
                Estabilidad y umbral operable
                <InfoTooltip text={ARBITRAGE_STABILITY_THRESHOLD_HELP} placement="bottom" />
              </h4>
              <span
                className="stability-panel__verdict-badge"
                style={{ color: noEstable ? COLORS.danger : COLORS.success }}
              >
                {noEstable ? "NO operable" : "Operable"}
              </span>
            </div>

            <p className="view-note">{ARBITRAGE_STABILITY_THRESHOLD_ONE_LINER}</p>

            {detail.estabilidad ? (
              <>
                <div className="pair-threshold-cell pair-threshold-cell--large">
                  <span className="pair-threshold-cell__label">
                    <span>
                      Este par cointegra{" "}
                      <strong
                        style={{
                          color: detail.estabilidad.fraccion_cointegrada >= stabilityThreshold ? COLORS.success : COLORS.danger,
                        }}
                      >
                        {formatPercent(detail.estabilidad.fraccion_cointegrada)}
                      </strong>{" "}
                      de las ventanas
                    </span>
                    <span className="pair-threshold-cell__target">tu umbral: {stabilityThresholdPct}%</span>
                  </span>
                  <div className="pair-threshold-cell__track">
                    <div
                      className="pair-threshold-cell__fill"
                      style={{
                        width: `${Math.min(100, detail.estabilidad.fraccion_cointegrada * 100)}%`,
                        background:
                          detail.estabilidad.fraccion_cointegrada >= stabilityThreshold ? COLORS.success : COLORS.danger,
                      }}
                    />
                    <div
                      className="pair-threshold-cell__threshold-marker"
                      style={{ left: `${stabilityThresholdPct}%` }}
                    />
                    <span
                      className="pair-threshold-cell__threshold-label"
                      style={{ left: `${stabilityThresholdPct}%` }}
                    >
                      tu umbral
                    </span>
                  </div>
                </div>

                <p className="stability-panel__verdict-line">
                  {formatPercent(detail.estabilidad.fraccion_cointegrada)}{" "}
                  {detail.estabilidad.fraccion_cointegrada >= stabilityThreshold ? "≥" : "<"} {stabilityThresholdPct}%
                  {" → "}
                  <strong style={{ color: noEstable ? COLORS.danger : COLORS.success }}>
                    {noEstable ? "NO operable" : "operable"}
                  </strong>
                </p>
              </>
            ) : (
              <p className="view-note">{detail.estabilidad_mensaje ?? "Sin dato de estabilidad rolling."}</p>
            )}

            <label className="pair-controls__field">
              Umbral de estabilidad (% ventanas cointegradas)
              <span className="pair-controls__field-value">Umbral: {stabilityThresholdPct}%</span>
              <input
                type="range"
                className="pair-controls__slider"
                min={0}
                max={1}
                step={0.05}
                value={stabilityThreshold}
                onChange={(event) => setStabilityThreshold(Number(event.target.value))}
              />
            </label>
          </div>

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
            6 — Backtest configurable
            <InfoTooltip text={ARBITRAGE_PAIR_BACKTEST_HELP} />
          </h3>
          {noEstable && <div className="honesty-banner">{ARBITRAGE_PAIR_BACKTEST_NOT_OPERABLE_WARNING}</div>}

          <div className="backtest-params">
            <label className="backtest-params__field">
              Umbral de entrada (|z|)
              <input
                type="number"
                min={0}
                step={0.1}
                className="backtest-params__input"
                value={btEntry}
                onChange={(event) => setBtEntry(Number(event.target.value))}
              />
            </label>
            <label className="backtest-params__field">
              Umbral de salida (|z|)
              <input
                type="number"
                min={0}
                step={0.1}
                className="backtest-params__input"
                value={btExit}
                onChange={(event) => setBtExit(Number(event.target.value))}
              />
            </label>
            <label className="backtest-params__field">
              Stop-loss (|z|)
              <input
                type="number"
                min={0}
                step={0.1}
                className="backtest-params__input"
                value={btStop}
                onChange={(event) => setBtStop(Number(event.target.value))}
              />
            </label>
            <label className="backtest-params__field">
              Costo de transacción (bps)
              <input
                type="number"
                min={0}
                step={1}
                className="backtest-params__input"
                value={btCostBps}
                onChange={(event) => setBtCostBps(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="panel-subtitle-row" style={{ marginTop: "var(--space-3)" }}>
            <label className={`toggle-chip${btLongOnly ? " toggle-chip--active" : ""}`}>
              <input type="checkbox" checked={btLongOnly} onChange={(event) => setBtLongOnly(event.target.checked)} />
              Long-only (rotación, sin ir en corto)
              <InfoTooltip text={ARBITRAGE_LONG_ONLY_HELP} placement="bottom" />
            </label>
          </div>

          <div className="metric-grid" style={{ marginTop: "var(--space-3)" }}>
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
            <MetricCard
              label="% tiempo fuera del mercado"
              value={`${(detail.backtest.metrics.pct_tiempo_fuera * 100).toFixed(1)}%`}
            />
          </div>
          <LineChartPanel series={backtestEquitySeries} height={280} />
          <p className="view-note">{ARBITRAGE_PAIR_BACKTEST_FLAT_PERIODS_NOTE}</p>
        </>
      )}
    </section>
  );
}
