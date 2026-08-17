/**
 * Panel de alertas en pantalla (Fase 8d): reglas simples evaluadas
 * CLIENT-SIDE contra los datos que ya trae `/api/studies` (Fase 8a) — sin
 * backend nuevo, sin polling propio (se reevalúan cuando React Query trae
 * datos frescos en `TechnicalAnalysisView`). Lógica de evaluación en
 * `../alerts.ts`, acá solo vive la UI.
 *
 * LIMITACIÓN HONESTA: notificaciones EN PANTALLA únicamente. Sin la app
 * abierta en el navegador no hay ninguna alerta — no hay notificaciones
 * push, email, ni nada que llegue si cerrás la pestaña. Las reglas sí
 * persisten en `localStorage` entre sesiones (Fase 8d).
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { OHLCVResponse, StudiesResponse } from "../types";
import { useLocalStorageState } from "../hooks/useLocalStorageState";
import {
  DEFAULT_THRESHOLD,
  RULE_LABELS,
  RULE_TYPES_WITH_THRESHOLD,
  describeRule,
  evaluateRule,
  type AlertRule,
  type AlertRuleType,
} from "../alerts";

const STORAGE_KEY = "cqd:alert-rules";
const TOAST_DURATION_MS = 8000;

interface ToastItem {
  id: string;
  message: string;
}

interface AlertsPanelProps {
  asset: string;
  interval: string;
  studies: StudiesResponse | undefined;
  ohlcv: OHLCVResponse | undefined;
}

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function AlertsPanel({ asset, interval, studies, ohlcv }: AlertsPanelProps) {
  const [allRules, setAllRules] = useLocalStorageState<AlertRule[]>(STORAGE_KEY, []);
  const [newType, setNewType] = useState<AlertRuleType>("rsi_above");
  const [newThreshold, setNewThreshold] = useState<number>(70);
  const [triggeredIds, setTriggeredIds] = useState<Set<string>>(new Set());
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const previouslyTriggeredRef = useRef<Set<string>>(new Set());

  const rules = useMemo(
    () => allRules.filter((rule) => rule.asset === asset && rule.interval === interval),
    [allRules, asset, interval],
  );

  // Se reevalúan todas las reglas del activo/timeframe actual cada vez que
  // llegan datos nuevos de /api/studies O cada vez que cambia la lista de
  // reglas (agregar/borrar una regla evalúa de inmediato contra los datos
  // YA cargados, sin esperar al próximo refetch de React Query). Solo se
  // dispara un toast en la TRANSICIÓN a "disparada" (no en cada
  // re-evaluación mientras se mantiene cumplida), comparando contra
  // `previouslyTriggeredRef`.
  useEffect(() => {
    if (!studies || !ohlcv) return;

    const nowTriggered = new Set<string>();
    for (const rule of rules) {
      if (evaluateRule(rule, studies, ohlcv)) {
        nowTriggered.add(rule.id);
      }
    }

    const newlyTriggered = [...nowTriggered].filter((id) => !previouslyTriggeredRef.current.has(id));
    if (newlyTriggered.length > 0) {
      const newToasts: ToastItem[] = newlyTriggered.map((id) => {
        const rule = rules.find((r) => r.id === id)!;
        return { id: createId(), message: `${describeRule(rule)} — ¡se cumplió!` };
      });
      setToasts((prev) => [...prev, ...newToasts]);
      newToasts.forEach((toast) => {
        setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== toast.id)), TOAST_DURATION_MS);
      });
    }

    previouslyTriggeredRef.current = nowTriggered;
    setTriggeredIds(nowTriggered);
  }, [studies, ohlcv, rules]);

  function handleTypeChange(type: AlertRuleType) {
    setNewType(type);
    if (DEFAULT_THRESHOLD[type] !== undefined) setNewThreshold(DEFAULT_THRESHOLD[type]);
  }

  function addRule() {
    const rule: AlertRule = {
      id: createId(),
      asset,
      interval,
      type: newType,
      threshold: RULE_TYPES_WITH_THRESHOLD.includes(newType) ? newThreshold : null,
      createdAt: new Date().toISOString(),
    };
    setAllRules((prev) => [...prev, rule]);
  }

  function removeRule(id: string) {
    setAllRules((prev) => prev.filter((rule) => rule.id !== id));
  }

  return (
    <div className="alerts-panel">
      <h3 className="panel-subtitle">
        Alertas ({asset}, {interval})
      </h3>
      <p className="view-note">
        Solo funcionan con la app abierta en esta vista — no hay notificaciones push externas.
      </p>

      <div className="alerts-panel__form">
        <select value={newType} onChange={(event) => handleTypeChange(event.target.value as AlertRuleType)}>
          {Object.entries(RULE_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
        {RULE_TYPES_WITH_THRESHOLD.includes(newType) && (
          <input
            type="number"
            value={newThreshold}
            onChange={(event) => setNewThreshold(Number(event.target.value))}
            className="alerts-panel__threshold"
          />
        )}
        <button type="button" className="secondary-button" onClick={addRule}>
          Agregar
        </button>
      </div>

      <ul className="alerts-panel__list">
        {rules.length === 0 && <li className="alerts-panel__empty">Sin alertas para este activo/timeframe.</li>}
        {rules.map((rule) => (
          <li
            key={rule.id}
            className={`alerts-panel__rule${triggeredIds.has(rule.id) ? " alerts-panel__rule--triggered" : ""}`}
          >
            <span>{describeRule(rule)}</span>
            <button type="button" className="icon-button" onClick={() => removeRule(rule.id)} aria-label="Eliminar alerta">
              ✕
            </button>
          </li>
        ))}
      </ul>

      <div className="toast-stack">
        {toasts.map((toast) => (
          <div key={toast.id} className="toast">
            {toast.message}
          </div>
        ))}
      </div>
    </div>
  );
}
