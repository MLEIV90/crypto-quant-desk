/**
 * Vista "Research" (Fase 8c) — consume `/api/prediction` (Fase 8c, sobre
 * `ml.models`). Equivalente web de la pestaña "Research (sin edge)" de la
 * app de escritorio (`app/widgets/prediction_panel.py`).
 *
 * INVESTIGACIÓN CON RESULTADO NEGATIVO, NO HERRAMIENTA OPERATIVA: el
 * modelo primario de ML nunca superó de forma consistente a los baselines
 * triviales en ninguna validación del proyecto (ver `ml/models.py`).
 *
 * `/api/prediction` es MUY LENTO (15-30s, entrena XGBoost con validación
 * purgeada) — a propósito NO se dispara solo al entrar a la vista
 * (`enabled: false`), el usuario lo pide con un botón explícito.
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError, getPrediction } from "../api";
import { MetricCard } from "../components/MetricCard";
import { StatusMessage } from "../components/StatusMessage";
import { DIRECTION_COLORS } from "../theme";

const HONESTY_TEXT =
  "INVESTIGACIÓN CON RESULTADO NEGATIVO, NO HERRAMIENTA OPERATIVA: este modelo de ML NUNCA superó de forma " +
  "consistente a los baselines triviales (azar / clase mayoritaria) en ninguna validación del proyecto. " +
  "Predicción OOS (out-of-sample, validación purgeada — nunca vio la fecha que predice) del modelo primario. " +
  "Para BTC/ETH usa features técnicas + on-chain (flujos de exchange, MVRV, direcciones activas); para " +
  "SOL/BNB/LTC, solo técnicas — sin cobertura on-chain suficiente en CoinMetrics. Se muestra tal cual sale, " +
  "sin maquillar — no es una recomendación de operar.";

interface ResearchViewProps {
  asset: string;
}

export function ResearchView({ asset }: ResearchViewProps) {
  const predictionQuery = useQuery({
    queryKey: ["prediction", asset],
    queryFn: () => getPrediction(asset),
    enabled: false,
    retry: 0,
  });

  const prediction = predictionQuery.data;
  const error = predictionQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  return (
    <section className="view">
      <div className="honesty-banner">{HONESTY_TEXT}</div>

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
            <MetricCard label="Accuracy purgeada" value={`${(prediction.accuracy_media * 100).toFixed(1)}%`} />
            <MetricCard label="Baseline azar" value={`${(prediction.baseline_azar * 100).toFixed(1)}%`} />
            <MetricCard label="Baseline mayoritaria" value={`${(prediction.baseline_mayoritaria * 100).toFixed(1)}%`} />
            <MetricCard label="ROC-AUC (ovr)" value={prediction.roc_auc_media.toFixed(3)} />
            <MetricCard
              label="¿Supera azar?"
              value={prediction.supera_azar ? "SÍ" : "NO"}
              valueColor={prediction.supera_azar ? undefined : "var(--color-danger)"}
            />
            <MetricCard
              label="¿Supera mayoritaria?"
              value={prediction.supera_mayoritaria ? "SÍ" : "NO"}
              valueColor={prediction.supera_mayoritaria ? undefined : "var(--color-danger)"}
            />
            <MetricCard
              label="On-chain usado"
              value={prediction.used_onchain ? `SÍ (${prediction.onchain_columns.length} cols.)` : "NO"}
            />
          </div>

          <h3 className="panel-subtitle">Top features</h3>
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Importancia</th>
              </tr>
            </thead>
            <tbody>
              {prediction.top_features.map(([name, importance]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{importance.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
