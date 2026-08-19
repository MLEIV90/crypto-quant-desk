/**
 * Vista "Arbitraje" (Fase 12b) — consume `/api/pairs/screening` y
 * `/api/pairs/detail` (sobre `pairs.cointegration`/`pairs.stability`/
 * `pairs.signals`, reutilizados tal cual, ver `api/main.py`).
 *
 * ENCUADRE HONESTO: esto es arbitraje ESTADÍSTICO (pairs trading), NO
 * arbitraje entre exchanges — ver `ARBITRAGE_INTRO_HELP`. La Fase 2 de este
 * proyecto ya mostró que la mayoría de los pares de `config.UNIVERSE` NO
 * están establemente cointegrados; esta vista expone ese resultado tal
 * cual, con su veredicto rojo/verde, en vez de esconderlo.
 *
 * El screening (tabla de arriba) siempre es DIARIO — `pairs.stability.screen_pairs_stability`
 * no soporta otro intervalo (ver `api/main.py::get_pairs_screening`) — así
 * que ahí no se usa el `interval` compartido del header. El detalle de un
 * par sí lo respeta de verdad.
 */

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getPairsDetail, getPairsScreening } from "../api";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type ReferenceLineSpec } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { StatusMessage } from "../components/StatusMessage";
import {
  ARBITRAGE_CONCEPTS_HELP,
  ARBITRAGE_INTRO_HELP,
  ARBITRAGE_NOT_OPERABLE_WARNING,
  ARBITRAGE_SCREENING_HELP,
} from "../helpTexts";
import { COLORS } from "../theme";

const SCREENING_INTERVAL = "1d";

const ZSCORE_REFERENCE_LINES: ReferenceLineSpec[] = [
  { price: 2, label: "+2", color: COLORS.danger },
  { price: 0, label: "0", color: COLORS.textMuted },
  { price: -2, label: "-2", color: COLORS.danger },
];

function formatPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(0)}%`;
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

  const screening = screeningQuery.data;
  const screeningError = screeningQuery.error;
  const screeningErrorMessage =
    screeningError instanceof ApiError ? screeningError.message : screeningError ? String(screeningError) : null;

  const detail = detailQuery.data;
  const detailError = detailQuery.error;
  const detailErrorMessage =
    detailError instanceof ApiError ? detailError.message : detailError ? String(detailError) : null;

  const zscoreSeries = useMemo(() => {
    if (!detail) return [];
    return [
      {
        id: "zscore",
        label: `z-score(${detail.asset_y}~${detail.asset_x})`,
        color: COLORS.accent,
        data: detail.fechas.map((fecha, i) => ({ time: fecha, value: detail.zscore[i] ?? null })),
      },
    ];
  }, [detail]);

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
                  <td>{formatPercent(fila.fraccion_cointegrada)}</td>
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
            />
            <MetricCard
              label="Half-life"
              value={formatHalfLife(detail.half_life_dias, detail.interval)}
              help={ARBITRAGE_CONCEPTS_HELP.halfLife}
            />
            <MetricCard
              label="% ventanas estables"
              value={detail.estabilidad ? formatPercent(detail.estabilidad.fraccion_cointegrada) : "Sin dato"}
              help={ARBITRAGE_CONCEPTS_HELP.estabilidad}
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

          <h3 className="panel-subtitle">
            Spread (z-score) con bandas ±2 / 0
            <InfoTooltip text={ARBITRAGE_CONCEPTS_HELP.spread} />
          </h3>
          <LineChartPanel series={zscoreSeries} height={320} referenceLines={ZSCORE_REFERENCE_LINES} />

          {detail.estabilidad_mensaje && (
            <p className="view-note">{detail.estabilidad_mensaje}</p>
          )}
        </>
      )}
    </section>
  );
}
