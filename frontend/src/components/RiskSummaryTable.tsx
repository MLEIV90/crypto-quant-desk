/**
 * Tabla compacta de riesgo actual de las 5 monedas (Fase 20b) — clic en
 * una fila cambia el activo activo, mismo patrón que `WatchlistPanel`.
 * Consume `/api/risk-summary`, RÁPIDO a propósito (sin ajustar GARCH, ver
 * `api/main.py::get_risk_summary`): el régimen de esta tabla se calcula
 * sobre volatilidad REALIZADA, no condicional GARCH como el resto de la
 * vista — útil para comparar entre monedas de un vistazo, no para
 * reemplazar el detalle GARCH del activo seleccionado (ver
 * `RISK_SUMMARY_HELP` para la explicación honesta de esa diferencia).
 */

import { REGIME_COLORS } from "../theme";
import type { RiskSummaryRow } from "../types";

interface RiskSummaryTableProps {
  filas: RiskSummaryRow[];
  activeAsset: string;
  onSelectAsset: (asset: string) => void;
}

function percentileLabel(value: number | null): string {
  return value === null ? "—" : `p${Math.round(value)}`;
}

export function RiskSummaryTable({ filas, activeAsset, onSelectAsset }: RiskSummaryTableProps) {
  // Fase 20c: el rótulo de base es el mismo para las 5 filas (es una
  // propiedad del MÉTODO de cálculo de esta tabla, no de cada moneda) —
  // se muestra una sola vez como pie de tabla en vez de repetirlo por fila.
  const first = filas[0];

  return (
    <>
      <table className="metrics-table risk-summary-table">
        <thead>
          <tr>
            <th>Moneda</th>
            <th>Vol. realizada</th>
            <th>Régimen</th>
            <th>VaR 95% actual</th>
          </tr>
        </thead>
        <tbody>
          {filas.map((fila) => {
            const isActive = fila.asset === activeAsset;
            return (
              <tr
                key={fila.asset}
                className={`risk-summary-table__row${isActive ? " risk-summary-table__row--active" : ""}`}
                onClick={() => onSelectAsset(fila.asset)}
                title={`Ver el detalle de riesgo de ${fila.asset}`}
              >
                <td className="risk-summary-table__asset">{fila.asset}</td>
                <td>
                  {(fila.vol_realizada * 100).toFixed(1)}%
                  <span className="risk-summary-table__percentil">{percentileLabel(fila.vol_realizada_percentil)}</span>
                </td>
                <td>
                  <span
                    className="risk-summary-table__regime"
                    style={fila.regimen ? { color: REGIME_COLORS[fila.regimen] } : undefined}
                  >
                    {fila.regimen ? fila.regimen.toUpperCase() : "—"}
                  </span>
                </td>
                <td>
                  {(fila.var95 * 100).toFixed(2)}%
                  <span className="risk-summary-table__percentil">{percentileLabel(fila.var95_percentil)}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {first && (
        <p className="view-note risk-summary-table__basis">
          Régimen: {first.regimen_basis} · VaR: {first.var95_basis}
        </p>
      )}
    </>
  );
}
