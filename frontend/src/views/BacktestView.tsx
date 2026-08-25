/**
 * Vista "Backtest" (Fase 8c, rehecha en Fase 21, pulida en Fase 23) —
 * consume `/api/backtest` (configurable: estrategia/costos/target_vol/rango
 * de fechas) y `/api/backtest-strategies` (catálogo con las descripciones,
 * para que el selector y los textos de esta vista nunca hardcodeen algo que
 * pueda desincronizarse de lo que el backend realmente calcula — mismo
 * patrón que los rótulos de base de `RiskView`, ver
 * `api/main.py::_BACKTEST_STRATEGY_CATALOG`).
 *
 * Fase 21 (rediseño): antes esta vista mostraba una única curva "Estrategia
 * vs. Buy&Hold" sin explicar qué estrategia era, sin parámetros, en escala
 * lineal y sin drawdown — un backtest que no explica la estrategia no
 * sirve. Ahora: (1) descripción de la estrategia elegida + su trade-off
 * explícito, (2) selector entre las estrategias que YA existen en
 * `signals.engine` (no hay ninguna nueva acá), (3) parámetros configurables
 * (costos, target_vol cuando aplica, rango de fechas), (4) gráfico
 * "underwater" de drawdown para ambas curvas, (5) toggle de escala
 * lineal/log en la curva de equity.
 *
 * Fase 23 (pulido, 7 hallazgos de revisión): log scale por DEFECTO (la
 * lineal aplasta la comparación), peor drawdown marcado en el gráfico
 * underwater, gráfico nuevo de EXPOSICIÓN (qué hizo la estrategia y
 * cuándo, no solo el resultado), métricas de actividad que sí distinguen
 * estrategias siempre-invertidas de las que salen del mercado, y la
 * sub-pestaña "Buy & hold puro" ya no se compara consigo misma.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getBacktest, getBacktestStrategies } from "../api";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel, type LineSeriesMarkerSpec } from "../components/LineChartPanel";
import { StatusMessage } from "../components/StatusMessage";
import {
  BACKTEST_DRAWDOWN_HELP,
  BACKTEST_EQUITY_HELP,
  BACKTEST_EXPOSURE_HELP,
  BACKTEST_LOG_SCALE_HELP,
  BACKTEST_METRIC_HELP,
  BACKTEST_PARAMS_HELP,
  BACKTEST_STRATEGY_SELECTOR_HELP,
} from "../helpTexts";
import { COLORS } from "../theme";
import type { EquityPoint } from "../types";

interface MetricRow {
  key: string;
  label: string;
  format: (value: number) => string;
}

const METRIC_ROWS: MetricRow[] = [
  { key: "cagr", label: "CAGR", format: (v) => `${(v * 100).toFixed(2)}%` },
  { key: "sharpe", label: "Sharpe", format: (v) => v.toFixed(2) },
  { key: "sortino", label: "Sortino", format: (v) => v.toFixed(2) },
  { key: "max_drawdown", label: "Max. drawdown", format: (v) => `${(v * 100).toFixed(2)}%` },
  { key: "calmar", label: "Calmar", format: (v) => v.toFixed(2) },
  { key: "exposicion_media", label: "Exposición promedio", format: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "pct_tiempo_fuera", label: "% tiempo fuera del mercado", format: (v) => `${(v * 100).toFixed(1)}%` },
  { key: "n_trades", label: "Cambios de dirección", format: (v) => v.toFixed(0) },
  { key: "turnover_total", label: "Turnover total", format: (v) => v.toFixed(2) },
];

// Fase 23 (mejora 2): marca el punto de PEOR drawdown de una curva
// underwater — sin esto, dos curvas que terminan cerca del cero (como en
// el momento actual del mercado) se ven "casi pegadas" y no se aprecia la
// diferencia real entre, p. ej., -71.8% y -13.8% en algún punto intermedio.
function worstPointMarker(curve: EquityPoint[], color: string): LineSeriesMarkerSpec[] {
  if (curve.length === 0) return [];
  let worst = curve[0];
  for (const point of curve) {
    if (point.valor < worst.valor) worst = point;
  }
  return [
    {
      time: worst.fecha,
      price: worst.valor,
      color,
      shape: "arrowDown",
      text: `${(worst.valor * 100).toFixed(1)}%`,
    },
  ];
}

interface BacktestViewProps {
  asset: string;
}

export function BacktestView({ asset }: BacktestViewProps) {
  const [strategy, setStrategy] = useState<string>("vol_targeting");
  const [costBps, setCostBps] = useState<number | undefined>(undefined);
  const [targetVol, setTargetVol] = useState<number | undefined>(undefined);
  const [fechaInicio, setFechaInicio] = useState<string | undefined>(undefined);
  const [fechaFin, setFechaFin] = useState<string | undefined>(undefined);
  // Fase 23 (mejora 3): arranca en escala LOGARÍTMICA — en lineal, el
  // crecimiento reciente aplasta visualmente la comparación y las curvas
  // se ven casi pegadas. El toggle sigue disponible para volver a lineal.
  const [logScale, setLogScale] = useState(true);

  const strategiesQuery = useQuery({ queryKey: ["backtest-strategies"], queryFn: getBacktestStrategies });
  const strategies = strategiesQuery.data?.estrategias ?? [];
  const selected = strategies.find((s) => s.id === strategy);

  // Precarga los defaults del backend apenas llegan (Fase 21): el usuario
  // arranca viendo con qué costo/target_vol se está corriendo de verdad,
  // no un campo vacío sin contexto.
  useEffect(() => {
    if (strategiesQuery.data && costBps === undefined) {
      setCostBps(strategiesQuery.data.cost_bps_default);
    }
  }, [strategiesQuery.data, costBps]);

  useEffect(() => {
    if (selected?.tiene_target_vol && selected.target_vol_default !== null && targetVol === undefined) {
      setTargetVol(selected.target_vol_default);
    }
  }, [selected, targetVol]);

  const backtestQuery = useQuery({
    queryKey: ["backtest", asset, strategy, costBps, targetVol, fechaInicio, fechaFin],
    queryFn: () =>
      getBacktest(asset, {
        strategy,
        costBps,
        targetVol: selected?.tiene_target_vol ? targetVol : undefined,
        fechaInicio,
        fechaFin,
      }),
    enabled: strategiesQuery.isSuccess,
  });
  const backtest = backtestQuery.data;
  const error = strategiesQuery.error ?? backtestQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  const estrategiaLabel = selected?.nombre ?? "Estrategia";
  // Fase 23 (mejora 7): cuando la estrategia elegida ES buy & hold, la
  // "comparación" contra el benchmark buy & hold es literalmente la misma
  // serie contra sí misma (dos líneas idénticas superpuestas) — se oculta
  // la mitad redundante en vez de dibujarla.
  const isBuyAndHold = strategy === "buy_and_hold";

  return (
    <section className="view">
      <p className="view-note">
        Los resultados incluyen costos de transacción. El desempeño pasado NO garantiza resultados futuros.
      </p>

      <h3 className="panel-subtitle">
        Qué estrategia se está probando
        <InfoTooltip text={BACKTEST_STRATEGY_SELECTOR_HELP} />
      </h3>
      <div className="period-selector">
        {strategies.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`period-selector__pill${s.id === strategy ? " period-selector__pill--active" : ""}`}
            onClick={() => {
              setStrategy(s.id);
              setTargetVol(undefined);
            }}
          >
            {s.nombre}
          </button>
        ))}
      </div>

      {strategiesQuery.isLoading && <StatusMessage kind="loading">Cargando estrategias disponibles…</StatusMessage>}

      {selected && (
        <div className="backtest-strategy-card">
          <p>{selected.descripcion}</p>
          <p className="backtest-strategy-card__objetivo">
            <strong>Objetivo:</strong> {selected.objetivo}
          </p>
          <p className="backtest-strategy-card__tradeoff">
            <strong>Lo que esto significa en la práctica:</strong> {selected.tradeoff}
          </p>
        </div>
      )}

      <h3 className="panel-subtitle">
        Parámetros
        <InfoTooltip text={BACKTEST_PARAMS_HELP} />
      </h3>
      <div className="backtest-params">
        <label className="backtest-params__field">
          Costo de transacción (bps)
          <input
            type="number"
            min={0}
            step={1}
            className="backtest-params__input"
            value={costBps ?? ""}
            onChange={(e) => setCostBps(e.target.value === "" ? undefined : Number(e.target.value))}
          />
        </label>

        {selected?.tiene_target_vol && (
          <label className="backtest-params__field">
            Volatilidad objetivo (anualizada)
            <input
              type="number"
              min={selected.target_vol_min ?? 0}
              max={selected.target_vol_max ?? undefined}
              step={0.05}
              className="backtest-params__input"
              value={targetVol ?? ""}
              onChange={(e) => setTargetVol(e.target.value === "" ? undefined : Number(e.target.value))}
            />
          </label>
        )}

        <label className="backtest-params__field">
          Desde
          <input
            type="date"
            className="backtest-params__input"
            value={fechaInicio ?? ""}
            onChange={(e) => setFechaInicio(e.target.value === "" ? undefined : e.target.value)}
          />
        </label>

        <label className="backtest-params__field">
          Hasta
          <input
            type="date"
            className="backtest-params__input"
            value={fechaFin ?? ""}
            onChange={(e) => setFechaFin(e.target.value === "" ? undefined : e.target.value)}
          />
        </label>
      </div>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && backtestQuery.isLoading && (
        <StatusMessage kind="loading">Corriendo backtest de {asset}…</StatusMessage>
      )}

      {!errorMessage && backtest && (
        <>
          {isBuyAndHold && (
            <p className="view-note">
              "Buy &amp; hold puro" ES la referencia contra la que se comparan las demás estrategias — no tiene
              sentido compararla consigo misma, así que acá se muestra sola, sin la columna/curva duplicada.
            </p>
          )}

          <div className="panel-subtitle-row">
            <h3 className="panel-subtitle">
              Equity: {estrategiaLabel}
              {!isBuyAndHold && " vs. buy & hold"}
              <InfoTooltip text={BACKTEST_EQUITY_HELP} />
            </h3>
            <label className="toggle-chip toggle-chip--active">
              <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} />
              Escala logarítmica
              <InfoTooltip text={BACKTEST_LOG_SCALE_HELP} />
            </label>
          </div>
          <LineChartPanel
            logScale={logScale}
            series={[
              {
                id: "estrategia",
                label: estrategiaLabel,
                color: COLORS.equityStrategy,
                data: backtest.equity_curve_estrategia.map((point) => ({ time: point.fecha, value: point.valor })),
              },
              ...(isBuyAndHold
                ? []
                : [
                    {
                      id: "buy-hold",
                      label: "Buy & hold",
                      color: COLORS.equityBuyHold,
                      data: backtest.equity_curve_buy_and_hold.map((point) => ({
                        time: point.fecha,
                        value: point.valor,
                      })),
                    },
                  ]),
            ]}
            height={320}
          />

          <h3 className="panel-subtitle">
            Drawdown (underwater): cuánto se está cayendo en cada momento
            <InfoTooltip text={BACKTEST_DRAWDOWN_HELP} />
          </h3>
          <LineChartPanel
            series={[
              {
                id: "drawdown-estrategia",
                label: estrategiaLabel,
                color: COLORS.equityStrategy,
                data: backtest.drawdown_curve_estrategia.map((point) => ({ time: point.fecha, value: point.valor })),
                markers: worstPointMarker(backtest.drawdown_curve_estrategia, COLORS.equityStrategy),
              },
              ...(isBuyAndHold
                ? []
                : [
                    {
                      id: "drawdown-buy-hold",
                      label: "Buy & hold",
                      color: COLORS.equityBuyHold,
                      data: backtest.drawdown_curve_buy_and_hold.map((point) => ({
                        time: point.fecha,
                        value: point.valor,
                      })),
                      markers: worstPointMarker(backtest.drawdown_curve_buy_and_hold, COLORS.equityBuyHold),
                    },
                  ]),
            ]}
            height={200}
          />

          <h3 className="panel-subtitle">
            Exposición: cuánto invertida está la estrategia en cada momento
            <InfoTooltip text={BACKTEST_EXPOSURE_HELP} />
          </h3>
          <LineChartPanel
            series={[
              {
                id: "exposicion",
                label: estrategiaLabel,
                color: COLORS.bollinger,
                data: backtest.exposure_curve_estrategia.map((point) => ({ time: point.fecha, value: point.valor })),
              },
            ]}
            referenceLines={[
              { price: 1, label: "100% largo", color: COLORS.textMuted },
              { price: 0, label: "afuera", color: COLORS.textMuted },
              { price: -1, label: "100% corto", color: COLORS.textMuted },
            ]}
            height={200}
          />

          <h3 className="panel-subtitle">Métricas lado a lado</h3>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Métrica</th>
                <th>{estrategiaLabel}</th>
                {!isBuyAndHold && <th>Buy &amp; hold</th>}
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map(({ key, label, format }) => (
                <tr key={key}>
                  <td>
                    {label}
                    <InfoTooltip text={BACKTEST_METRIC_HELP[key]} />
                  </td>
                  <td>{format(backtest.metrics_estrategia[key])}</td>
                  {!isBuyAndHold && <td>{format(backtest.metrics_buy_and_hold[key])}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
