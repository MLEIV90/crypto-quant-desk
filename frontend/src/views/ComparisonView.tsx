/**
 * Vista "Comparación" (Fase 12a, enriquecida en Fase 27) — consume
 * `/api/compare` (Fase 12a, sobre `analysis.comparison.compare_assets`,
 * reutilizado tal cual; Fase 27 le agrega riesgo del período y la fecha
 * base). Compara el rendimiento normalizado (base 100) de varias monedas
 * elegidas en un mismo gráfico, más un ranking del período con
 * rendimiento Y riesgo — no solo "quién subió más".
 *
 * A diferencia del resto de las vistas, acá el "activo activo" del header
 * NO se usa — el multi-select de monedas es estado PROPIO de esta vista
 * (`selected`), independiente de `asset` en `App.tsx`. Sí reutiliza el
 * `interval` compartido (1d/1h, mismo criterio que la vista Estadística en
 * Fase 11) y el `PeriodSelector` ya existente (Fase 10a) para la ventana
 * de comparación.
 *
 * Fase 27: (1) escala logarítmica por DEFECTO en el gráfico — en lineal, la
 * moneda que más subió aplasta visualmente a las demás; (2) el ranking
 * ahora tiene 4 columnas ordenables (rendimiento/vol/drawdown/Sharpe), no
 * solo rendimiento; (3) se muestra la fecha base de la comparación
 * explícitamente; (4) un texto honesto conecta rendimiento alto con riesgo
 * alto.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getCompare } from "../api";
import { AssetMultiSelect } from "../components/AssetMultiSelect";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type LineSeriesSpec } from "../components/LineChartPanel";
import { candleLimitForPeriod, DEFAULT_PERIOD, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { StatusMessage } from "../components/StatusMessage";
import {
  COMPARISON_INTRO_HELP,
  COMPARISON_LOG_SCALE_HELP,
  COMPARISON_RANKING_HELP,
  COMPARISON_RISK_HONEST_TEXT,
} from "../helpTexts";
import { colorForAsset, COLORS } from "../theme";

const DEFAULT_SELECTED_ASSETS: string[] = ["BTC", "ETH"];

type SortKey = "rendimiento" | "vol" | "drawdown" | "sharpe";

const SORT_COLUMN_LABELS: Record<SortKey, string> = {
  rendimiento: "Rendimiento del período",
  vol: "Vol. anualizada",
  drawdown: "Máx. drawdown",
  sharpe: "Sharpe",
};

function toggleInSet<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

/** `rendimiento_total_pct` ya viene en PUNTOS PORCENTUALES desde el backend
 * (p. ej. 457.2 == +457.2%); `vol_anualizada`/`max_drawdown` vienen en
 * escala DECIMAL (0.57 == 57%) — dos formatters distintos a propósito, ver
 * el mismo patrón en `StatisticsView.tsx`.
 */
function formatScaledPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatDecimalPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString();
}

interface RankingRow {
  asset: string;
  rendimiento: number;
  vol: number;
  drawdown: number;
  sharpe: number;
}

interface ComparisonViewProps {
  assets: string[];
  interval: string;
}

export function ComparisonView({ assets, interval }: ComparisonViewProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set(DEFAULT_SELECTED_ASSETS));
  const [period, setPeriod] = useState<PeriodKey>(DEFAULT_PERIOD);
  const candleLimit = candleLimitForPeriod(period, interval);
  // Fase 27 (mejora 1): arranca en LOG — la lineal aplasta visualmente a la
  // moneda que menos subió cuando otra subió muchas veces más.
  const [logScale, setLogScale] = useState(true);
  const [sortBy, setSortBy] = useState<SortKey>("rendimiento");
  const [sortDesc, setSortDesc] = useState(true);

  const selectedList = useMemo(() => assets.filter((asset) => selected.has(asset)), [assets, selected]);
  const assetsParam = selectedList.join(",");

  const compareQuery = useQuery({
    queryKey: ["compare", assetsParam, interval, candleLimit],
    queryFn: () => getCompare(assetsParam, interval, candleLimit),
    enabled: selectedList.length > 0,
  });

  const compare = compareQuery.data;
  const error = compareQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  const series: LineSeriesSpec[] = useMemo(() => {
    if (!compare) return [];
    return compare.assets.map((asset, index) => ({
      id: asset,
      label: asset,
      color: colorForAsset(asset, index),
      data: compare.fechas.map((fecha, i) => ({ time: fecha, value: compare.series[asset][i] ?? null })),
    }));
  }, [compare]);

  const rankingRows: RankingRow[] = useMemo(() => {
    if (!compare) return [];
    return compare.assets
      .filter((asset) => asset in compare.riesgo)
      .map((asset) => ({
        asset,
        rendimiento: compare.rendimiento_total_pct[asset],
        vol: compare.riesgo[asset].vol_anualizada,
        drawdown: compare.riesgo[asset].max_drawdown,
        sharpe: compare.riesgo[asset].sharpe,
      }));
  }, [compare]);

  const sortedRanking = useMemo(() => {
    const sorted = [...rankingRows].sort((a, b) => (sortDesc ? b[sortBy] - a[sortBy] : a[sortBy] - b[sortBy]));
    return sorted;
  }, [rankingRows, sortBy, sortDesc]);

  function handleSort(key: SortKey) {
    if (key === sortBy) {
      setSortDesc((prev) => !prev);
    } else {
      setSortBy(key);
      setSortDesc(true);
    }
  }

  function sortIndicator(key: SortKey): string {
    if (key !== sortBy) return "";
    return sortDesc ? " ▼" : " ▲";
  }

  return (
    <section className="view">
      <p className="view-note">{COMPARISON_INTRO_HELP}</p>

      <fieldset className="toggles">
        <legend>Monedas a comparar</legend>
        <AssetMultiSelect
          assets={assets}
          selected={selected}
          onToggle={(asset) => setSelected((prev) => toggleInSet(prev, asset))}
        />
      </fieldset>

      <PeriodSelector active={period} onChange={setPeriod} />

      {selectedList.length === 0 && (
        <StatusMessage kind="error">Elegí al menos una moneda para comparar.</StatusMessage>
      )}
      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && selectedList.length > 0 && compareQuery.isLoading && (
        <StatusMessage kind="loading">Comparando {selectedList.join(", ")}…</StatusMessage>
      )}

      {!errorMessage && compare && series.length > 0 && (
        <>
          {compare.fecha_base && (
            <p className="view-note">
              Comparando desde {formatDate(compare.fecha_base)} (primera fecha con datos de todas las monedas
              elegidas).
            </p>
          )}

          <div className="comparison-legend">
            {series.map((spec) => (
              <span key={spec.id} className="comparison-legend__item">
                <span className="comparison-legend__dot" style={{ background: spec.color }} />
                {spec.label}
              </span>
            ))}
          </div>

          <div className="panel-subtitle-row">
            <span />
            <label className="toggle-chip toggle-chip--active">
              <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
              Escala logarítmica
              <InfoTooltip text={COMPARISON_LOG_SCALE_HELP} />
            </label>
          </div>
          <LineChartPanel series={series} height={420} logScale={logScale} />

          <h3 className="panel-subtitle">
            Ranking del período: rendimiento y riesgo
            <InfoTooltip text={COMPARISON_RANKING_HELP} />
          </h3>
          <p className="view-note">{COMPARISON_RISK_HONEST_TEXT}</p>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Moneda</th>
                {(Object.keys(SORT_COLUMN_LABELS) as SortKey[]).map((key) => (
                  <th key={key} onClick={() => handleSort(key)} style={{ cursor: "pointer" }}>
                    {SORT_COLUMN_LABELS[key]}
                    {sortIndicator(key)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedRanking.map((row) => (
                <tr key={row.asset}>
                  <td>{row.asset}</td>
                  <td style={{ color: row.rendimiento >= 0 ? COLORS.success : COLORS.danger }}>
                    {formatScaledPercent(row.rendimiento)}
                  </td>
                  <td>{formatDecimalPercent(row.vol)}</td>
                  <td style={{ color: COLORS.danger }}>{formatDecimalPercent(row.drawdown)}</td>
                  <td style={{ color: row.sharpe >= 0 ? COLORS.success : COLORS.danger }}>
                    {row.sharpe.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
