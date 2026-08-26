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
import { ApiError, getBacktest, getGarchSeries, getOhlcv, getOhlcvCsv, getRisk, getRiskSummary } from "../api";
import { CsvDownloadButton } from "../components/CsvDownloadButton";
import { InfoTooltip } from "../components/InfoTooltip";
import { LineChartPanel } from "../components/LineChartPanel";
import { MetricCard } from "../components/MetricCard";
import { candleLimitForPeriod, PeriodSelector, type PeriodKey } from "../components/PeriodSelector";
import { RegimeStrip } from "../components/RegimeStrip";
import { RiskSummaryTable } from "../components/RiskSummaryTable";
import { ReturnHistogramChart } from "../components/ReturnHistogramChart";
import { StatusMessage } from "../components/StatusMessage";
import {
  RISK_ACTUAL_VS_HISTORICO_HELP,
  RISK_HISTOGRAM_HELP,
  RISK_INTRO_HELP,
  RISK_METRIC_HELP,
  RISK_PERCENTILE_HELP,
  RISK_REGIME_STRIP_HELP,
  RISK_SUMMARY_HELP,
  RISK_VOL_TARGETING_HELP,
} from "../helpTexts";
import { COLORS, DIRECTION_COLORS, REGIME_COLORS } from "../theme";

const RISK_INTERVAL = "1d"; // el modelo GARCH del proyecto es diario, ver api/main.py::DEFAULT_RISK_INTERVAL

// Fase 20a/20c: umbrales de color del percentil — deliberadamente los
// mismos ~70/90 que usa REGIME_COLORS de forma implícita (percentiles de
// volatilidad muy altos son justamente lo que dispara "tensión" en
// models.garch.volatility_regime), para que el color de la tarjeta no
// contradiga el del régimen mostrado al lado. `base` (Fase 20c) viene de
// la API (`risk.percentiles.base`) — nunca hardcodeado acá, para que el
// texto nunca pueda desincronizarse de contra qué ventana se comparó de
// verdad (ver api/models.py::RiskPercentiles).
function percentileDescriptor(percentile: number | null, base: string): { text: string; color?: string } {
  if (percentile === null) {
    return { text: `Sin historia suficiente (${base}) para calcular el percentil.` };
  }
  const rounded = Math.round(percentile);
  const text = `percentil ${rounded} vs. ${base} — más alto que el ${rounded}% de ese período`;
  if (percentile >= 90) return { text, color: COLORS.danger };
  if (percentile >= 70) return { text, color: COLORS.warning };
  return { text };
}

interface RiskViewProps {
  asset: string;
  onAssetChange: (asset: string) => void;
}

export function RiskView({ asset, onAssetChange }: RiskViewProps) {
  const [pricePeriod, setPricePeriod] = useState<PeriodKey>("todo");
  const priceCandleLimit = candleLimitForPeriod(pricePeriod, RISK_INTERVAL);

  const riskQuery = useQuery({ queryKey: ["risk", asset], queryFn: () => getRisk(asset) });
  const garchQuery = useQuery({ queryKey: ["garch-series", asset], queryFn: () => getGarchSeries(asset) });
  const priceQuery = useQuery({
    queryKey: ["ohlcv", asset, RISK_INTERVAL, "risk-view", priceCandleLimit],
    queryFn: () => getOhlcv(asset, RISK_INTERVAL, priceCandleLimit),
  });
  // Fase 20b: sin `asset` en la queryKey a propósito — las 5 filas son
  // siempre las mismas 5 monedas, no dependen de cuál está seleccionada;
  // cambiar de activo (incluso haciendo clic en esta misma tabla) no debe
  // disparar un refetch de esto.
  const riskSummaryQuery = useQuery({ queryKey: ["risk-summary"], queryFn: getRiskSummary });
  // Fase 22: dos backtests, no uno — antes esta sección pedía el default
  // "combo" (dirección del engine + vol targeting) y le atribuía el
  // resultado a "vol targeting" a secas. Ahora se piden las DOS estrategias
  // por separado para poder mostrar la descomposición real (ver más abajo):
  // cuánto aporta el sizing por volatilidad SOLO vs. cuánto aporta
  // combinarlo con la señal direccional. Ambas devuelven el mismo
  // buy & hold (mismo activo/costos), se usa el de `comboQuery` para la
  // comparación.
  const backtestVolTargetingQuery = useQuery({
    queryKey: ["backtest", asset, "vol_targeting"],
    queryFn: () => getBacktest(asset, { strategy: "vol_targeting" }),
  });
  const backtestComboQuery = useQuery({
    queryKey: ["backtest", asset, "combo"],
    queryFn: () => getBacktest(asset, { strategy: "combo" }),
  });

  const risk = riskQuery.data;
  const garch = garchQuery.data;
  const price = priceQuery.data;
  const riskSummary = riskSummaryQuery.data;
  const backtestVolTargeting = backtestVolTargetingQuery.data;
  const backtestCombo = backtestComboQuery.data;

  const isLoading = riskQuery.isLoading || garchQuery.isLoading || priceQuery.isLoading;
  const error = riskQuery.error ?? garchQuery.error ?? priceQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <p className="view-note">{RISK_INTRO_HELP}</p>

      <h3 className="panel-subtitle">
        Riesgo actual de las 5 monedas
        <InfoTooltip text={RISK_SUMMARY_HELP} />
      </h3>
      {riskSummaryQuery.isLoading && <StatusMessage kind="loading">Calculando riesgo de las 5 monedas…</StatusMessage>}
      {riskSummaryQuery.error && (
        <StatusMessage kind="error">
          {riskSummaryQuery.error instanceof ApiError ? riskSummaryQuery.error.message : String(riskSummaryQuery.error)}
        </StatusMessage>
      )}
      {riskSummary && (
        <RiskSummaryTable filas={riskSummary.filas} activeAsset={asset} onSelectAsset={onAssetChange} />
      )}

      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}
      {!errorMessage && isLoading && (
        <StatusMessage kind="loading">Ajustando modelo GARCH para {asset}… (puede tardar unos segundos)</StatusMessage>
      )}

      {!errorMessage && risk && (
        <>
          <div className="metric-grid">
            <MetricCard
              label="Vol. realizada anualizada"
              value={`${(risk.vol_realizada * 100).toFixed(2)}%`}
              help={`${RISK_METRIC_HELP.volRealizada} ${RISK_PERCENTILE_HELP}`}
              subtext={percentileDescriptor(risk.percentiles.vol_realizada, risk.percentiles.base).text}
              subtextColor={percentileDescriptor(risk.percentiles.vol_realizada, risk.percentiles.base).color}
            />
            <MetricCard label="Modelo GARCH ganador" value={risk.modelo_garch} help={RISK_METRIC_HELP.modeloGarch} />
            <MetricCard
              label="Vol. condicional GARCH"
              value={`${(risk.vol_garch * 100).toFixed(2)}%`}
              help={`${RISK_METRIC_HELP.volGarch} ${RISK_PERCENTILE_HELP}`}
              subtext={percentileDescriptor(risk.percentiles.vol_garch, risk.percentiles.base).text}
              subtextColor={percentileDescriptor(risk.percentiles.vol_garch, risk.percentiles.base).color}
            />
            <MetricCard
              label="Régimen de volatilidad"
              value={risk.regimen ? risk.regimen.toUpperCase() : "—"}
              valueColor={risk.regimen ? REGIME_COLORS[risk.regimen] : undefined}
              help={`${RISK_METRIC_HELP.regimen} Base: ${risk.regimen_basis}.`}
              subtext={`vs. ${risk.regimen_basis}`}
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

          <h3 className="panel-subtitle">
            VaR / Expected Shortfall: actual vs. histórico
            <InfoTooltip text={RISK_ACTUAL_VS_HISTORICO_HELP} />
          </h3>
          <p className="view-note">
            El VaR/ES "actual" (empírico, último año) refleja el régimen de HOY — es el MISMO método y el mismo
            número que la tabla de las 5 monedas de más arriba. El "histórico" es un único número sobre toda la
            serie y NO cambia aunque el mercado esté en calma o en tensión — mirá el actual para saber cuánto
            riesgo hay ahora.
          </p>
          <div className="metric-grid">
            <MetricCard
              label="VaR 95% actual (empírico, último año)"
              value={`${(risk.var95_actual * 100).toFixed(2)}%`}
              help={`${RISK_METRIC_HELP.var95} ${RISK_PERCENTILE_HELP} Base: ${risk.actual_basis}.`}
              subtext={percentileDescriptor(risk.percentiles.var95, risk.percentiles.base).text}
              subtextColor={percentileDescriptor(risk.percentiles.var95, risk.percentiles.base).color}
            />
            <MetricCard
              label="ES 95% actual (empírico, último año)"
              value={`${(risk.es95_actual * 100).toFixed(2)}%`}
              help={`${RISK_METRIC_HELP.es95} ${RISK_PERCENTILE_HELP} Base: ${risk.actual_basis}.`}
              subtext={percentileDescriptor(risk.percentiles.es95, risk.percentiles.base).text}
              subtextColor={percentileDescriptor(risk.percentiles.es95, risk.percentiles.base).color}
            />
            <MetricCard
              label="VaR 95% histórico (toda la serie)"
              value={`${(risk.var95 * 100).toFixed(2)}%`}
              help={RISK_METRIC_HELP.var95}
              subtext={`referencia: ${risk.historico_basis} (no refleja el régimen actual)`}
            />
            <MetricCard
              label="ES 95% histórico (toda la serie)"
              value={`${(risk.es95 * 100).toFixed(2)}%`}
              help={RISK_METRIC_HELP.es95}
              subtext={`referencia: ${risk.historico_basis} (no refleja el régimen actual)`}
            />
          </div>
        </>
      )}

      {!errorMessage && risk && (
        <>
          <h3 className="panel-subtitle">
            Distribución de retornos diarios ({asset})
          </h3>
          <p className="view-note">{RISK_HISTOGRAM_HELP}</p>
          <ReturnHistogramChart
            binEdges={risk.histograma.bin_edges}
            counts={risk.histograma.counts}
            var95Return={risk.histograma.var95_return}
            es95Return={risk.histograma.es95_return}
          />
        </>
      )}

      <h3 className="panel-subtitle">
        El valor de gestionar el riesgo: qué aporta cada pieza
        <InfoTooltip text={RISK_VOL_TARGETING_HELP} />
      </h3>
      <p className="view-note">
        El vol targeting SOLO (siempre comprado, solo ajusta el tamaño según la volatilidad) reduce el
        drawdown de forma modesta. El salto grande viene de COMBINARLO con la señal direccional del
        engine, que además se sale del mercado en las malas — se muestran las tres para que se vea qué
        aporta cada pieza, no solo el resultado final combinado.
      </p>
      {(backtestVolTargetingQuery.isLoading || backtestComboQuery.isLoading) && (
        <StatusMessage kind="loading">Corriendo backtest de {asset} (buy & hold, vol targeting y combinado)…</StatusMessage>
      )}
      {(backtestVolTargetingQuery.error || backtestComboQuery.error) && (
        <StatusMessage kind="error">
          {(() => {
            const err = backtestVolTargetingQuery.error ?? backtestComboQuery.error;
            return err instanceof ApiError ? err.message : String(err);
          })()}
        </StatusMessage>
      )}
      {backtestVolTargeting && backtestCombo && (
        <>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Métrica</th>
                <th>Buy &amp; hold</th>
                <th>Vol targeting (solo)</th>
                <th>Engine + vol targeting</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Máx. drawdown</td>
                <td style={{ color: COLORS.danger }}>
                  {(backtestVolTargeting.metrics_buy_and_hold.max_drawdown * 100).toFixed(1)}%
                </td>
                <td
                  style={{
                    color:
                      backtestVolTargeting.metrics_estrategia.max_drawdown >
                      backtestVolTargeting.metrics_buy_and_hold.max_drawdown
                        ? COLORS.success
                        : COLORS.danger,
                  }}
                >
                  {(backtestVolTargeting.metrics_estrategia.max_drawdown * 100).toFixed(1)}%
                </td>
                <td
                  style={{
                    color:
                      backtestCombo.metrics_estrategia.max_drawdown > backtestCombo.metrics_buy_and_hold.max_drawdown
                        ? COLORS.success
                        : COLORS.danger,
                  }}
                >
                  {(backtestCombo.metrics_estrategia.max_drawdown * 100).toFixed(1)}%
                </td>
              </tr>
              <tr>
                <td>Sharpe</td>
                <td>{backtestVolTargeting.metrics_buy_and_hold.sharpe.toFixed(2)}</td>
                <td
                  style={{
                    color:
                      backtestVolTargeting.metrics_estrategia.sharpe >= backtestVolTargeting.metrics_buy_and_hold.sharpe
                        ? COLORS.success
                        : undefined,
                  }}
                >
                  {backtestVolTargeting.metrics_estrategia.sharpe.toFixed(2)}
                </td>
                <td
                  style={{
                    color:
                      backtestCombo.metrics_estrategia.sharpe >= backtestCombo.metrics_buy_and_hold.sharpe
                        ? COLORS.success
                        : undefined,
                  }}
                >
                  {backtestCombo.metrics_estrategia.sharpe.toFixed(2)}
                </td>
              </tr>
            </tbody>
          </table>
          <LineChartPanel
            series={[
              {
                id: "buy-hold",
                label: "Buy & hold",
                color: COLORS.equityBuyHold,
                data: backtestCombo.equity_curve_buy_and_hold.map((point) => ({ time: point.fecha, value: point.valor })),
              },
              {
                id: "vol-targeting",
                label: "Vol targeting (solo)",
                color: COLORS.equityStrategy,
                data: backtestVolTargeting.equity_curve_estrategia.map((point) => ({ time: point.fecha, value: point.valor })),
              },
              {
                id: "combo",
                label: "Engine + vol targeting",
                color: COLORS.bollinger,
                data: backtestCombo.equity_curve_estrategia.map((point) => ({ time: point.fecha, value: point.valor })),
              },
            ]}
            height={320}
          />
        </>
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
          <h3 className="panel-subtitle">Régimen de volatilidad en el tiempo</h3>
          <p className="view-note">{RISK_REGIME_STRIP_HELP}</p>
          <RegimeStrip fechas={garch.fechas} regimenes={garch.regimen_serie} />
        </>
      )}
    </section>
  );
}
