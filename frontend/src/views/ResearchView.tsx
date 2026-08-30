/**
 * Vista "Research" (Fase 8c, rehecha en Fase 24) — vitrina de TODA la
 * investigación predictiva del proyecto, no solo el modelo de ML. Consume
 * `/api/prediction` (Fase 8c, ML supervisado on-demand, MUY LENTO) y
 * `/api/research-experiments` (Fase 24, LEE resultados YA guardados de
 * Deep RL y rotación por momentum — rápido, un archivo JSON chico).
 *
 * INVESTIGACIÓN CON RESULTADO NEGATIVO, NO HERRAMIENTA OPERATIVA: los TRES
 * enfoques documentados acá (ML supervisado, Deep RL, rotación por
 * momentum) fallan en encontrar una ventaja predictiva consistente sobre
 * baselines triviales — ver `RESEARCH_THESIS_TEXT`/`RESEARCH_SYNTHESIS_TEXT`
 * en `helpTexts.ts`. Ese resultado ES el hallazgo: se presenta con el mismo
 * rigor que un resultado positivo, sin maquillar ni sugerir que sirve para
 * operar.
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError, getPrediction, getResearchExperiments } from "../api";
import { BarChart, type BarChartDatum } from "../components/BarChart";
import { InfoTooltip } from "../components/InfoTooltip";
import { MetricCard } from "../components/MetricCard";
import { StatusMessage } from "../components/StatusMessage";
import {
  RESEARCH_ML_APPROACH_HELP,
  RESEARCH_METRIC_HELP,
  RESEARCH_RL_APPROACH_HELP,
  RESEARCH_RL_TABLE_HELP,
  RESEARCH_ROTATION_APPROACH_HELP,
  RESEARCH_ROTATION_METRIC_HELP,
  RESEARCH_ROTATION_TABLE_HELP,
  RESEARCH_SYNTHESIS_TEXT,
  RESEARCH_THESIS_TEXT,
} from "../helpTexts";
import { COLORS, DIRECTION_COLORS } from "../theme";
import type { RlResearchResult, RotationResearchResult } from "../types";

interface ResearchViewProps {
  asset: string;
}

function fechaExperimento(iso: string): string {
  return new Date(iso).toLocaleDateString("es-AR", { year: "numeric", month: "long", day: "numeric" });
}

function VeredictoBadge({ ok, textoOk, textoNo }: { ok: boolean; textoOk: string; textoNo: string }) {
  return <span className={`pair-semaphore pair-semaphore--${ok ? "ok" : "no"}`}>{ok ? textoOk : textoNo}</span>;
}

function RlApproachCard({ rl }: { rl: RlResearchResult | null }) {
  if (!rl) {
    return (
      <div className="backtest-strategy-card">
        <p>{RESEARCH_RL_APPROACH_HELP}</p>
        <p className="backtest-strategy-card__tradeoff">Experimento no corrido aún.</p>
      </div>
    );
  }

  const baselineRows = rl.summary_table.filter((row) => row.estrategia !== "RL (PPO)");
  const rlRow = rl.summary_table.find((row) => row.estrategia === "RL (PPO)");

  return (
    <div className="backtest-strategy-card">
      <p>{RESEARCH_RL_APPROACH_HELP}</p>
      <p className="backtest-strategy-card__objetivo">
        <strong>Cómo se validó:</strong> walk-forward con {rl.blocks.length} bloques, {rl.params.seeds.length}{" "}
        semillas por bloque ({rl.n_ppo_runs} corridas de PPO en total), {rl.params.cost_bps} bps de costo, ventana
        mínima de entrenamiento de {rl.params.min_train_days} días. Rango OOS: {rl.oos_date_range[0].slice(0, 10)} a{" "}
        {rl.oos_date_range[1].slice(0, 10)}.
      </p>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Estrategia</th>
            <th>Sharpe OOS (media ± std)</th>
          </tr>
        </thead>
        <tbody>
          {rlRow && (
            <tr>
              <td>{rlRow.estrategia}</td>
              <td>
                {rlRow.sharpe_media.toFixed(2)} ± {rlRow.sharpe_std.toFixed(2)}
              </td>
            </tr>
          )}
          {baselineRows.map((row) => (
            <tr key={row.estrategia}>
              <td>{row.estrategia}</td>
              <td>
                {row.sharpe_media.toFixed(2)}
                {row.sharpe_std > 0 ? ` ± ${row.sharpe_std.toFixed(2)}` : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="backtest-strategy-card__tradeoff">
        <InfoTooltip text={RESEARCH_RL_TABLE_HELP} />{" "}
        <strong>Veredicto:</strong>{" "}
        <VeredictoBadge
          ok={rl.conclusion.supera_a_todos_los_baselines_consistentemente}
          textoOk="SUPERA A LOS BASELINES"
          textoNo="SIN EDGE CONSISTENTE"
        />{" "}
        — peor semilla Sharpe {rl.conclusion.sharpe_rl_peor_semilla.toFixed(2)}, mejor semilla{" "}
        {rl.conclusion.sharpe_rl_mejor_semilla.toFixed(2)} (buy &amp; hold BTC: Sharpe{" "}
        {rl.conclusion.sharpe_baselines["buy_hold_btc"]?.toFixed(2) ?? "—"}).
      </p>
      <p className="view-note">Experimento del {fechaExperimento(rl.fecha_experimento)}.</p>
    </div>
  );
}

function RotationApproachCard({ rotation }: { rotation: RotationResearchResult | null }) {
  if (!rotation) {
    return (
      <div className="backtest-strategy-card">
        <p>{RESEARCH_ROTATION_APPROACH_HELP}</p>
        <p className="backtest-strategy-card__tradeoff">Experimento no corrido aún.</p>
      </div>
    );
  }

  const pares = Object.entries(rotation.per_pair_robusto);

  return (
    <div className="backtest-strategy-card">
      <p>{RESEARCH_ROTATION_APPROACH_HELP}</p>
      <p className="backtest-strategy-card__objetivo">
        <strong>Cómo se validó:</strong> {rotation.params.lookback_grid.length} ventanas de lookback ×{" "}
        {rotation.params.rebalance_grid.length} frecuencias de rebalanceo × {rotation.params.pairs.length} pares ={" "}
        {rotation.n_combos} combinaciones, cada una contra sus propios baselines (buy &amp; hold de cada moneda,
        50/50 rebalanceado).
      </p>
      <div className="metric-grid">
        <MetricCard
          label="Pares robustos"
          value={`${rotation.conclusion.pares_robustos.length} de ${pares.length}`}
          subtext={`${(rotation.conclusion.fraccion_pares_robustos * 100).toFixed(0)}% — consistente con azar`}
          help={RESEARCH_ROTATION_METRIC_HELP.pares_robustos}
        />
        <MetricCard
          label={`Par principal (${rotation.conclusion.par_principal})`}
          value={rotation.conclusion.robusto_par_principal ? "Robusto" : "No robusto"}
          valueColor={rotation.conclusion.robusto_par_principal ? undefined : COLORS.danger}
          help={RESEARCH_ROTATION_METRIC_HELP.par_principal}
        />
      </div>
      <p className="backtest-strategy-card__tradeoff">
        <InfoTooltip text={RESEARCH_ROTATION_TABLE_HELP} />{" "}
        <strong>Veredicto:</strong>{" "}
        <VeredictoBadge ok={rotation.conclusion.veredicto_global} textoOk="ROBUSTO" textoNo="NO ROBUSTO" />
        {rotation.conclusion.pares_robustos.length > 0 && (
          <> — únicos pares robustos: {rotation.conclusion.pares_robustos.join(", ")}.</>
        )}
      </p>
      <p className="view-note">Experimento del {fechaExperimento(rotation.fecha_experimento)}.</p>
    </div>
  );
}

export function ResearchView({ asset }: ResearchViewProps) {
  const predictionQuery = useQuery({
    queryKey: ["prediction", asset],
    queryFn: () => getPrediction(asset),
    enabled: false,
    retry: 0,
  });
  // Fase 24: a diferencia de la predicción de ML, esto SOLO lee un archivo
  // JSON chico ya guardado — rápido, se puede pedir apenas se entra a la vista.
  const experimentsQuery = useQuery({ queryKey: ["research-experiments"], queryFn: getResearchExperiments });

  const prediction = predictionQuery.data;
  const error = predictionQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  const accuracyBarData: BarChartDatum[] = prediction
    ? [
        { label: "Modelo", value: prediction.accuracy_media, title: "Accuracy OOS purgeada del modelo" },
        { label: "Baseline azar", value: prediction.baseline_azar, title: "Accuracy de elegir al azar" },
        {
          label: "Baseline mayoritaria",
          value: prediction.baseline_mayoritaria,
          title: "Accuracy de predecir siempre la clase más frecuente",
        },
      ]
    : [];

  const featuresBarData: BarChartDatum[] = prediction
    ? prediction.top_features.map(([name, importance]) => ({ label: name, value: importance }))
    : [];

  return (
    <section className="view">
      <p className="view-note">{RESEARCH_THESIS_TEXT}</p>

      <h3 className="panel-subtitle">Los enfoques que probamos</h3>
      {experimentsQuery.isLoading && <StatusMessage kind="loading">Cargando experimentos guardados…</StatusMessage>}
      {experimentsQuery.error && (
        <StatusMessage kind="error">
          {experimentsQuery.error instanceof ApiError ? experimentsQuery.error.message : String(experimentsQuery.error)}
        </StatusMessage>
      )}

      <div className="backtest-strategy-card">
        <p>{RESEARCH_ML_APPROACH_HELP}</p>
        <p className="backtest-strategy-card__tradeoff">
          <strong>Veredicto:</strong> <VeredictoBadge ok={false} textoOk="—" textoNo="SIN EDGE" /> en ninguna
          validación del proyecto — corré la predicción para {asset} más abajo para ver el resultado actualizado.
        </p>
      </div>

      {experimentsQuery.data && (
        <>
          <RlApproachCard rl={experimentsQuery.data.rl} />
          <RotationApproachCard rotation={experimentsQuery.data.rotation} />
        </>
      )}

      <h3 className="panel-subtitle">ML supervisado: probalo vos mismo</h3>
      <div className="research-controls">
        <button
          type="button"
          className="primary-button"
          onClick={() => void predictionQuery.refetch()}
          disabled={predictionQuery.isFetching}
        >
          {predictionQuery.isFetching ? "Corriendo…" : `Correr predicción para ${asset}`}
        </button>
        <p className="view-note">
          Cómputo pesado (entrena un modelo con validación purgeada): puede tardar entre 15 y 30 segundos.
        </p>
      </div>

      {predictionQuery.isFetching && (
        <StatusMessage kind="loading">Entrenando/evaluando el modelo para {asset}…</StatusMessage>
      )}
      {errorMessage && <StatusMessage kind="error">{errorMessage}</StatusMessage>}

      {prediction && (
        <>
          <div className="suggester-panel__suggestion" style={{ color: DIRECTION_COLORS[prediction.prediccion_clase] }}>
            {prediction.prediccion_clase}{" "}
            <span className="suggester-panel__confidence">({(prediction.prediccion_confianza * 100).toFixed(1)}% conf.)</span>
          </div>
          <p className="view-note">Última fecha con datos: {new Date(prediction.ultima_fecha).toLocaleDateString()}</p>

          <div className="metric-grid">
            <MetricCard
              label="Accuracy purgeada"
              value={`${(prediction.accuracy_media * 100).toFixed(1)}%`}
              help={RESEARCH_METRIC_HELP.accuracy_media}
            />
            <MetricCard
              label="Baseline azar"
              value={`${(prediction.baseline_azar * 100).toFixed(1)}%`}
              help={RESEARCH_METRIC_HELP.baseline_azar}
            />
            <MetricCard
              label="Baseline mayoritaria"
              value={`${(prediction.baseline_mayoritaria * 100).toFixed(1)}%`}
              help={RESEARCH_METRIC_HELP.baseline_mayoritaria}
            />
            <MetricCard
              label="ROC-AUC (ovr)"
              value={prediction.roc_auc_media.toFixed(3)}
              help={RESEARCH_METRIC_HELP.roc_auc_media}
            />
            <MetricCard
              label="¿Supera azar?"
              value={prediction.supera_azar ? "SÍ" : "NO"}
              valueColor={prediction.supera_azar ? undefined : COLORS.danger}
              help={RESEARCH_METRIC_HELP.supera_azar}
            />
            <MetricCard
              label="¿Supera mayoritaria?"
              value={prediction.supera_mayoritaria ? "SÍ" : "NO"}
              valueColor={prediction.supera_mayoritaria ? undefined : COLORS.danger}
              help={RESEARCH_METRIC_HELP.supera_mayoritaria}
            />
            <MetricCard
              label="On-chain usado"
              value={prediction.used_onchain ? `SÍ (${prediction.onchain_columns.length} cols.)` : "NO"}
              help={RESEARCH_METRIC_HELP.used_onchain}
            />
          </div>

          <h4 className="panel-subtitle">Accuracy vs. baselines</h4>
          <BarChart horizontal data={accuracyBarData} formatValue={(v) => `${(v * 100).toFixed(1)}%`} />

          <h4 className="panel-subtitle">Importancia de features (top {prediction.top_features.length})</h4>
          <BarChart horizontal data={featuresBarData} formatValue={(v) => v.toFixed(3)} />
        </>
      )}

      <div className="honesty-banner">{RESEARCH_SYNTHESIS_TEXT}</div>
    </section>
  );
}
