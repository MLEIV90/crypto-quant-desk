/**
 * Vista "Backtest" (Fase 8c) — consume `/api/backtest` (Fase 8a, sobre
 * `signals.engine.generate_positions` + `backtest.engine`). Equivalente
 * web de la pestaña "Backtest" de la app de escritorio
 * (`app/widgets/backtest_panel.py`).
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError, getBacktest } from "../api";
import { LineChartPanel } from "../components/LineChartPanel";
import { StatusMessage } from "../components/StatusMessage";
import { COLORS } from "../theme";

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
  { key: "n_trades", label: "N° de trades", format: (v) => v.toFixed(0) },
  { key: "turnover_total", label: "Turnover total", format: (v) => v.toFixed(2) },
];

interface BacktestViewProps {
  asset: string;
}

export function BacktestView({ asset }: BacktestViewProps) {
  const backtestQuery = useQuery({ queryKey: ["backtest", asset], queryFn: () => getBacktest(asset) });
  const backtest = backtestQuery.data;
  const error = backtestQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <p className="view-note">
        Los resultados incluyen costos de transacción. El desempeño pasado NO garantiza resultados futuros.
      </p>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && backtestQuery.isLoading && (
        <StatusMessage kind="loading">Corriendo backtest de {asset}…</StatusMessage>
      )}

      {!errorMessage && backtest && (
        <>
          <h3 className="panel-subtitle">Equity: estrategia vs. buy &amp; hold</h3>
          <LineChartPanel
            series={[
              {
                id: "estrategia",
                label: "Estrategia (engine)",
                color: COLORS.equityStrategy,
                data: backtest.equity_curve_estrategia.map((point) => ({ time: point.fecha, value: point.valor })),
              },
              {
                id: "buy-hold",
                label: "Buy & hold",
                color: COLORS.equityBuyHold,
                data: backtest.equity_curve_buy_and_hold.map((point) => ({ time: point.fecha, value: point.valor })),
              },
            ]}
            height={360}
          />

          <h3 className="panel-subtitle">Métricas lado a lado</h3>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Métrica</th>
                <th>Estrategia</th>
                <th>Buy &amp; hold</th>
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map(({ key, label, format }) => (
                <tr key={key}>
                  <td>{label}</td>
                  <td>{format(backtest.metrics_estrategia[key])}</td>
                  <td>{format(backtest.metrics_buy_and_hold[key])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
