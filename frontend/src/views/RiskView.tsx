/**
 * Vista "Riesgo" (Fase 8c) — consume `/api/risk` y `/api/garch-series`
 * (Fase 8a, sobre `models.garch`/`metrics.risk_measures`/`signals.engine`).
 * Equivalente web de la pestaña "Riesgo" de la app de escritorio
 * (`app/widgets/risk_panel.py`).
 *
 * LENTO: ambos endpoints ajustan un modelo GARCH (ver `api/main.py`) —
 * el estado de carga lo avisa explícitamente, no se queda "pensando" en
 * silencio.
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError, getGarchSeries, getOhlcv, getRisk } from "../api";
import { LineChartPanel } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { StatusMessage } from "../components/StatusMessage";
import { COLORS, DIRECTION_COLORS, REGIME_COLORS } from "../theme";

const PRICE_CANDLE_LIMIT = 300;

interface RiskViewProps {
  asset: string;
}

export function RiskView({ asset }: RiskViewProps) {
  const riskQuery = useQuery({ queryKey: ["risk", asset], queryFn: () => getRisk(asset) });
  const garchQuery = useQuery({ queryKey: ["garch-series", asset], queryFn: () => getGarchSeries(asset) });
  const priceQuery = useQuery({
    queryKey: ["ohlcv", asset, "1d", "risk-view"],
    queryFn: () => getOhlcv(asset, "1d", PRICE_CANDLE_LIMIT),
  });

  const risk = riskQuery.data;
  const garch = garchQuery.data;
  const price = priceQuery.data;

  const isLoading = riskQuery.isLoading || garchQuery.isLoading || priceQuery.isLoading;
  const error = riskQuery.error ?? garchQuery.error ?? priceQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <p className="view-note">
        Estas son medidas de RIESGO y de SIZING calculadas sobre la historia — no son una predicción de precio.
      </p>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && isLoading && (
        <StatusMessage kind="loading">Ajustando modelo GARCH para {asset}… (puede tardar unos segundos)</StatusMessage>
      )}

      {!errorMessage && risk && (
        <div className="metric-grid">
          <MetricCard label="Vol. realizada anualizada" value={`${(risk.vol_realizada * 100).toFixed(2)}%`} />
          <MetricCard label="Modelo GARCH ganador" value={risk.modelo_garch} />
          <MetricCard label="Vol. condicional GARCH" value={`${(risk.vol_garch * 100).toFixed(2)}%`} />
          <MetricCard
            label="Régimen de volatilidad"
            value={risk.regimen ? risk.regimen.toUpperCase() : "—"}
            valueColor={risk.regimen ? REGIME_COLORS[risk.regimen] : undefined}
          />
          <MetricCard label="VaR 95% (pérdida diaria)" value={`${(risk.var95 * 100).toFixed(2)}%`} />
          <MetricCard label="Expected Shortfall 95%" value={`${(risk.es95 * 100).toFixed(2)}%`} />
          <MetricCard
            label="Señal del engine"
            value={`${risk.accion} (score ${risk.score >= 0 ? "+" : ""}${risk.score.toFixed(2)})`}
            valueColor={DIRECTION_COLORS[risk.accion]}
          />
          <MetricCard
            label="Tamaño sugerido (vol targeting)"
            value={`${risk.tamano_sugerido >= 0 ? "+" : ""}${risk.tamano_sugerido.toFixed(2)}x`}
          />
        </div>
      )}

      {!errorMessage && price && (
        <>
          <h3 className="panel-subtitle">Precio ({asset})</h3>
          <LineChartPanel
            series={[
              {
                id: "price",
                label: `${asset} close`,
                color: COLORS.accent,
                data: price.velas.map((vela) => ({ time: vela.fecha, value: vela.close })),
              },
            ]}
            height={280}
          />
        </>
      )}

      {!errorMessage && garch && (
        <>
          <h3 className="panel-subtitle">Volatilidad condicional GARCH ({garch.modelo_garch})</h3>
          <LineChartPanel
            series={[
              {
                id: "garch-vol",
                label: "Vol. condicional anualizada",
                color: COLORS.rsi,
                data: garch.fechas.map((fecha, index) => ({ time: fecha, value: garch.vol_condicional[index] })),
              },
            ]}
            height={280}
          />
        </>
      )}
    </section>
  );
}
