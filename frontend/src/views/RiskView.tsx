/**
 * Vista "Riesgo" (Fase 8c) — consume `/api/risk` y `/api/garch-series`
 * (Fase 8a, sobre `models.garch`/`metrics.risk_measures`/`signals.engine`).
 * Equivalente web de la pestaña "Riesgo" de la app de escritorio
 * (`app/widgets/risk_panel.py`).
 *
 * LENTO: ambos endpoints ajustan un modelo GARCH (ver `api/main.py`) —
 * el estado de carga lo avisa explícitamente, no se queda "pensando" en
 * silencio.
 *
 * Fase 10a: el gráfico de precio tenía un `PRICE_CANDLE_LIMIT` fijo (300
 * velas) — ahora usa su PROPIO `PeriodSelector` (independiente del de
 * Análisis Técnico, porque esta vista siempre opera en diario), default
 * "Todo" para que se vea el histórico completo junto a la volatilidad
 * GARCH (que ya se grafica entera, sin límite, más abajo).
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, getGarchSeries, getOhlcv, getOhlcvCsv, getRisk } from "../api";
import { CsvDownloadButton } from "../components/CsvDownloadButton";
import { LineChartPanel } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { candleLimitForPeriod, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { StatusMessage } from "../components/StatusMessage";
import { RISK_INTRO_HELP, RISK_METRIC_HELP } from "../helpTexts";
import { COLORS, DIRECTION_COLORS, REGIME_COLORS } from "../theme";

const RISK_INTERVAL = "1d"; // el modelo GARCH del proyecto es diario, ver api/main.py::DEFAULT_RISK_INTERVAL

interface RiskViewProps {
  asset: string;
}

export function RiskView({ asset }: RiskViewProps) {
  const [pricePeriod, setPricePeriod] = useState<PeriodKey>("todo");
  const priceCandleLimit = candleLimitForPeriod(pricePeriod, RISK_INTERVAL);

  const riskQuery = useQuery({ queryKey: ["risk", asset], queryFn: () => getRisk(asset) });
  const garchQuery = useQuery({ queryKey: ["garch-series", asset], queryFn: () => getGarchSeries(asset) });
  const priceQuery = useQuery({
    queryKey: ["ohlcv", asset, RISK_INTERVAL, "risk-view", priceCandleLimit],
    queryFn: () => getOhlcv(asset, RISK_INTERVAL, priceCandleLimit),
  });

  const risk = riskQuery.data;
  const garch = garchQuery.data;
  const price = priceQuery.data;

  const isLoading = riskQuery.isLoading || garchQuery.isLoading || priceQuery.isLoading;
  const error = riskQuery.error ?? garchQuery.error ?? priceQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <p className="view-note">{RISK_INTRO_HELP}</p>

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && isLoading && (
        <StatusMessage kind="loading">Ajustando modelo GARCH para {asset}… (puede tardar unos segundos)</StatusMessage>
      )}

      {!errorMessage && risk && (
        <div className="metric-grid">
          <MetricCard
            label="Vol. realizada anualizada"
            value={`${(risk.vol_realizada * 100).toFixed(2)}%`}
            help={RISK_METRIC_HELP.volRealizada}
          />
          <MetricCard label="Modelo GARCH ganador" value={risk.modelo_garch} help={RISK_METRIC_HELP.modeloGarch} />
          <MetricCard
            label="Vol. condicional GARCH"
            value={`${(risk.vol_garch * 100).toFixed(2)}%`}
            help={RISK_METRIC_HELP.volGarch}
          />
          <MetricCard
            label="Régimen de volatilidad"
            value={risk.regimen ? risk.regimen.toUpperCase() : "—"}
            valueColor={risk.regimen ? REGIME_COLORS[risk.regimen] : undefined}
            help={RISK_METRIC_HELP.regimen}
          />
          <MetricCard
            label="VaR 95% (pérdida diaria)"
            value={`${(risk.var95 * 100).toFixed(2)}%`}
            help={RISK_METRIC_HELP.var95}
          />
          <MetricCard
            label="Expected Shortfall 95%"
            value={`${(risk.es95 * 100).toFixed(2)}%`}
            help={RISK_METRIC_HELP.es95}
          />
          <MetricCard
            label="Señal del engine"
            value={`${risk.accion} (score ${risk.score >= 0 ? "+" : ""}${risk.score.toFixed(2)})`}
            valueColor={DIRECTION_COLORS[risk.accion]}
            help={RISK_METRIC_HELP.senal}
          />
          <MetricCard
            label="Tamaño sugerido (vol targeting)"
            value={`${risk.tamano_sugerido >= 0 ? "+" : ""}${risk.tamano_sugerido.toFixed(2)}x`}
            help={RISK_METRIC_HELP.sizing}
          />
        </div>
      )}

      {!errorMessage && price && (
        <>
          <div className="panel-subtitle-row">
            <h3 className="panel-subtitle">Precio ({asset})</h3>
            <PeriodSelector active={pricePeriod} onChange={setPricePeriod} />
            <CsvDownloadButton
              label="Descargar CSV"
              filename={`ohlcv_${asset}_${RISK_INTERVAL}.csv`}
              fetchCsv={() => getOhlcvCsv(asset, RISK_INTERVAL, priceCandleLimit)}
              queryKey={["export-ohlcv-csv", asset, RISK_INTERVAL, priceCandleLimit]}
            />
          </div>
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
