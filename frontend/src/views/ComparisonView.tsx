/**
 * Vista "Comparación" (Fase 12a) — consume `/api/compare` (Fase 12a, sobre
 * `analysis.comparison.compare_assets`, reutilizado tal cual). Compara el
 * rendimiento normalizado (base 100) de varias monedas elegidas en un
 * mismo gráfico, más un ranking del período.
 *
 * A diferencia del resto de las vistas, acá el "activo activo" del header
 * NO se usa — el multi-select de monedas es estado PROPIO de esta vista
 * (`selected`), independiente de `asset` en `App.tsx`. Sí reutiliza el
 * `interval` compartido (1d/1h, mismo criterio que la vista Estadística en
 * Fase 11) y el `PeriodSelector` ya existente (Fase 10a) para la ventana
 * de comparación.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getCompare } from "../api";
import { AssetMultiSelect } from "../components/AssetMultiSelect";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type LineSeriesSpec } from "../components/LineChartPanel";
import { candleLimitForPeriod, DEFAULT_PERIOD, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { StatusMessage } from "../components/StatusMessage";
import { COMPARISON_INTRO_HELP, COMPARISON_RANKING_HELP } from "../helpTexts";
import { colorForAsset } from "../theme";

const DEFAULT_SELECTED_ASSETS: string[] = ["BTC", "ETH"];

function toggleInSet<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

interface ComparisonViewProps {
  assets: string[];
  interval: string;
}

export function ComparisonView({ assets, interval }: ComparisonViewProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set(DEFAULT_SELECTED_ASSETS));
  const [period, setPeriod] = useState<PeriodKey>(DEFAULT_PERIOD);
  const candleLimit = candleLimitForPeriod(period, interval);

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

  const ranking = useMemo(() => {
    if (!compare) return [];
    return Object.entries(compare.rendimiento_total_pct).sort((a, b) => b[1] - a[1]);
  }, [compare]);

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
          <div className="comparison-legend">
            {series.map((spec) => (
              <span key={spec.id} className="comparison-legend__item">
                <span className="comparison-legend__dot" style={{ background: spec.color }} />
                {spec.label}
              </span>
            ))}
          </div>
          <LineChartPanel series={series} height={420} />

          <h3 className="panel-subtitle">
            Ranking del período
            <InfoTooltip text={COMPARISON_RANKING_HELP} />
          </h3>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Moneda</th>
                <th>Rendimiento del período</th>
              </tr>
            </thead>
            <tbody>
              {ranking.map(([asset, pct]) => (
                <tr key={asset}>
                  <td>{asset}</td>
                  <td style={{ color: pct >= 0 ? "var(--color-success)" : "var(--color-danger)" }}>
                    {formatPercent(pct)}
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
