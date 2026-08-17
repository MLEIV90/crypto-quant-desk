/**
 * Panel lateral del sugeridor de consenso (Fase 8c) — consume
 * `/api/suggester` (Fase 8a, sobre `signals.suggester.suggest`, Fase 7a).
 * Solo dibuja el `SuggesterResponse` que le pasan; no calcula nada, no
 * llama a la API directamente (eso lo hace `views/TechnicalAnalysisView.tsx`).
 *
 * HONESTIDAD: el desempeño histórico de la regla (`desempeno_historico`)
 * SIEMPRE se muestra, destacado, junto a la sugerencia — nunca se omite,
 * igual que en el backend (`signals/suggester.py`) y en la app de
 * escritorio (`app/widgets/suggester_panel.py`).
 */

import type { SuggesterResponse } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { SUGGESTER_HELP } from "../helpTexts";
import { DIRECTION_COLORS, VOTE_COLORS } from "../theme";

interface SuggesterPanelProps {
  data: SuggesterResponse;
}

const STUDY_LABELS: Record<string, string> = {
  rsi: "RSI",
  medias: "Medias móviles",
  macd: "MACD",
  estocastico: "Estocástico",
  bollinger: "Bollinger",
  pivotes: "Pivote",
};

const VOTE_LABELS: Record<string, string> = {
  alcista: "▲ alcista",
  bajista: "▼ bajista",
  neutral: "— neutral",
};

function formatPercent(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)}%`;
}

export function SuggesterPanel({ data }: SuggesterPanelProps) {
  const color = DIRECTION_COLORS[data.sugerencia] ?? DIRECTION_COLORS.ESPERAR;
  const desempeno = data.desempeno_historico;

  return (
    <aside className="suggester-panel">
      <h2 className="panel-title">
        Sugeridor de consenso
        <InfoTooltip text={SUGGESTER_HELP} />
      </h2>

      <div className="suggester-panel__suggestion" style={{ color }}>
        {data.sugerencia}
      </div>
      <p className="suggester-panel__confidence">Confianza: {(data.confianza * 100).toFixed(0)}%</p>
      <p className="suggester-panel__votes">
        Votos — alcistas: {data.votos_alcistas} · bajistas: {data.votos_bajistas} · neutrales: {data.votos_neutrales}
      </p>

      <h3 className="panel-subtitle">Detalle por estudio</h3>
      <dl className="vote-list">
        {Object.entries(data.detalle).map(([key, vote]) => (
          <div key={key} className="vote-list__row">
            <dt>{STUDY_LABELS[key] ?? key}</dt>
            <dd style={{ color: VOTE_COLORS[vote] }}>{VOTE_LABELS[vote] ?? vote}</dd>
          </div>
        ))}
      </dl>

      <div className="performance-highlight">
        <p>
          Esta sugerencia es un <strong>CONSENSO DE INDICADORES</strong> — no una señal con edge demostrado. Su
          desempeño histórico medido:
        </p>
        <ul>
          <li>
            Sharpe {desempeno.sharpe_sugeridor.toFixed(2)} vs. buy &amp; hold {desempeno.sharpe_buy_and_hold.toFixed(2)}
          </li>
          <li>
            CAGR {formatPercent(desempeno.cagr_sugeridor)} vs. buy &amp; hold {formatPercent(desempeno.cagr_buy_and_hold)}
          </li>
          <li>
            Max. drawdown {formatPercent(desempeno.max_drawdown_sugeridor)} vs. buy &amp; hold{" "}
            {formatPercent(desempeno.max_drawdown_buy_and_hold)}
          </li>
        </ul>
        <p className="performance-highlight__cta">Usala como APOYO a tu análisis, NO como orden de operar.</p>
      </div>
    </aside>
  );
}
