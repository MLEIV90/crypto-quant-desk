/**
 * Panel de alertas en pantalla (Fase 8d, ampliado en Fase 13c): reglas
 * evaluadas CLIENT-SIDE contra los datos que ya traen `/api/studies`/
 * `/api/ohlcv`/`/api/volume-profile` — sin backend nuevo, sin polling
 * propio (se reevalúan cuando React Query trae datos frescos en
 * `TechnicalAnalysisView`). Lógica de evaluación en `../alerts.ts`, acá
 * solo vive la UI.
 *
 * Fase 13c agrega: más tipos de regla (ver `../alerts.ts`), sonido
 * opcional (`../sound.ts`), historial persistido, activar/desactivar sin
 * borrar, edición de reglas existentes, y reglas para CUALQUIER moneda del
 * universo (no solo la que está en pantalla).
 *
 * LIMITACIÓN HONESTA: notificaciones EN PANTALLA únicamente. Sin la app
 * abierta en el navegador no hay ninguna alerta — no hay notificaciones
 * push, email, ni nada que llegue si cerrás la pestaña. Una regla para un
 * activo distinto al cargado en pantalla queda guardada pero NO se evalúa
 * hasta que cambies a verlo (ver el badge "en vivo" de cada regla). Las
 * reglas/historial/preferencia de sonido sí persisten en `localStorage`
 * entre sesiones.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { OHLCVResponse, StudiesResponse, VolumeProfileResponse } from "../types";
import { useLocalStorageState } from "../hooks/useLocalStorageState";
import { InfoTooltip } from "./InfoTooltip";
import { ALERTS_HONESTY_HELP } from "../helpTexts";
import { playAlertBeep } from "../sound";
import {
  DEFAULT_THRESHOLD,
  MA_LABELS,
  RULE_LABELS,
  RULE_TYPES_WITH_LEVEL_INPUT,
  RULE_TYPES_WITH_MA,
  RULE_TYPES_WITH_THRESHOLD,
  describeRule,
  evaluateRule,
  type AlertHistoryEntry,
  type AlertRule,
  type AlertRuleType,
  type MaKey,
} from "../alerts";

const RULES_STORAGE_KEY = "cqd:alert-rules";
const HISTORY_STORAGE_KEY = "cqd:alert-history";
const SOUND_STORAGE_KEY = "cqd:alert-sound-enabled";
const TOAST_DURATION_MS = 8000;
const MAX_HISTORY = 50;

interface ToastItem {
  id: string;
  message: string;
}

interface AlertsPanelProps {
  asset: string;
  interval: string;
  assets: string[];
  studies: StudiesResponse | undefined;
  ohlcv: OHLCVResponse | undefined;
  volumeProfile?: VolumeProfileResponse | null;
  onAlertTriggered?: (asset: string) => void;
}

function createId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function AlertsPanel({ asset, interval, assets, studies, ohlcv, volumeProfile, onAlertTriggered }: AlertsPanelProps) {
  const [allRules, setAllRules] = useLocalStorageState<AlertRule[]>(RULES_STORAGE_KEY, []);
  const [history, setHistory] = useLocalStorageState<AlertHistoryEntry[]>(HISTORY_STORAGE_KEY, []);
  const [soundEnabled, setSoundEnabled] = useLocalStorageState<boolean>(SOUND_STORAGE_KEY, true);

  const [newAsset, setNewAsset] = useState(asset);
  const [newType, setNewType] = useState<AlertRuleType>("rsi_above");
  const [newThreshold, setNewThreshold] = useState<number>(70);
  const [newMaKey, setNewMaKey] = useState<MaKey | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);

  const [triggeredIds, setTriggeredIds] = useState<Set<string>>(new Set());
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const previouslyTriggeredRef = useRef<Set<string>>(new Set());

  // Solo las reglas del activo/timeframe CARGADO EN PANTALLA se pueden
  // evaluar en vivo (ver la limitación arquitectural del docstring) — el
  // resto de `allRules` se lista igual (gestión: editar/activar/borrar),
  // solo que sin evaluar.
  const liveRules = useMemo(
    () => allRules.filter((rule) => rule.asset === asset && rule.interval === interval),
    [allRules, asset, interval],
  );

  const sortedRules = useMemo(
    () => [...allRules].sort((a, b) => a.asset.localeCompare(b.asset) || a.createdAt.localeCompare(b.createdAt)),
    [allRules],
  );

  useEffect(() => {
    if (!studies || !ohlcv) return;

    const nowTriggered = new Set<string>();
    for (const rule of liveRules) {
      if (evaluateRule(rule, studies, ohlcv, volumeProfile)) {
        nowTriggered.add(rule.id);
      }
    }

    const newlyTriggered = [...nowTriggered].filter((id) => !previouslyTriggeredRef.current.has(id));
    if (newlyTriggered.length > 0) {
      const timestamp = new Date().toISOString();
      const newToasts: ToastItem[] = [];
      const newHistoryEntries: AlertHistoryEntry[] = [];

      newlyTriggered.forEach((id) => {
        const rule = liveRules.find((r) => r.id === id);
        if (!rule) return;
        const message = `${describeRule(rule)} — ¡se cumplió!`;
        newToasts.push({ id: createId(), message });
        newHistoryEntries.push({ id: createId(), ruleId: rule.id, asset: rule.asset, interval: rule.interval, message, timestamp });
      });

      setToasts((prev) => [...prev, ...newToasts]);
      newToasts.forEach((toast) => {
        setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== toast.id)), TOAST_DURATION_MS);
      });
      setHistory((prev) => [...newHistoryEntries, ...prev].slice(0, MAX_HISTORY));
      if (soundEnabled) playAlertBeep();
      onAlertTriggered?.(asset);
    }

    previouslyTriggeredRef.current = nowTriggered;
    setTriggeredIds(nowTriggered);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [studies, ohlcv, volumeProfile, liveRules, soundEnabled]);

  function resetForm() {
    setEditingId(null);
    setNewAsset(asset);
    setNewType("rsi_above");
    setNewThreshold(70);
    setNewMaKey(null);
  }

  function handleTypeChange(type: AlertRuleType) {
    setNewType(type);
    if (DEFAULT_THRESHOLD[type] !== undefined) setNewThreshold(DEFAULT_THRESHOLD[type]!);
    if (RULE_TYPES_WITH_MA.includes(type)) {
      setNewMaKey((prev) => prev ?? "sma_20");
    } else {
      setNewMaKey(null);
    }
  }

  function submitForm() {
    const threshold = RULE_TYPES_WITH_THRESHOLD.includes(newType) ? newThreshold : null;
    const maKey = RULE_TYPES_WITH_MA.includes(newType) ? (newMaKey ?? "sma_20") : null;

    if (editingId) {
      setAllRules((prev) =>
        prev.map((rule) => (rule.id === editingId ? { ...rule, asset: newAsset, type: newType, threshold, maKey } : rule)),
      );
    } else {
      const rule: AlertRule = {
        id: createId(),
        asset: newAsset,
        interval,
        type: newType,
        threshold,
        maKey,
        enabled: true,
        createdAt: new Date().toISOString(),
      };
      setAllRules((prev) => [...prev, rule]);
    }
    resetForm();
  }

  function startEdit(rule: AlertRule) {
    setEditingId(rule.id);
    setNewAsset(rule.asset);
    setNewType(rule.type);
    setNewThreshold(rule.threshold ?? DEFAULT_THRESHOLD[rule.type] ?? 0);
    setNewMaKey(rule.maKey);
  }

  function removeRule(id: string) {
    setAllRules((prev) => prev.filter((rule) => rule.id !== id));
    if (editingId === id) resetForm();
  }

  function toggleEnabled(id: string) {
    setAllRules((prev) => prev.map((rule) => (rule.id === id ? { ...rule, enabled: !rule.enabled } : rule)));
  }

  return (
    <div className="alerts-panel">
      <div className="alerts-panel__header-row">
        <h3 className="panel-subtitle">
          Alertas
          <InfoTooltip text={ALERTS_HONESTY_HELP} placement="bottom" />
        </h3>
        <button
          type="button"
          className={`secondary-button${soundEnabled ? " secondary-button--active" : ""}`}
          onClick={() => setSoundEnabled((prev) => !prev)}
          title={soundEnabled ? "Silenciar sonido de alertas" : "Activar sonido de alertas"}
        >
          {soundEnabled ? "🔊 Sonido" : "🔇 Sonido"}
        </button>
      </div>
      <p className="view-note">
        Mirando {asset} ({interval}): solo las reglas de ese activo/timeframe están "en vivo" ahora — el resto queda
        guardado pero no se evalúa hasta que cambies a verlo. Sin notificaciones push externas.
      </p>

      <div className="alerts-panel__form">
        <select value={newAsset} onChange={(event) => setNewAsset(event.target.value)}>
          {assets.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select value={newType} onChange={(event) => handleTypeChange(event.target.value as AlertRuleType)}>
          {Object.entries(RULE_LABELS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
        {RULE_TYPES_WITH_MA.includes(newType) && (
          <select value={newMaKey ?? "sma_20"} onChange={(event) => setNewMaKey(event.target.value as MaKey)}>
            {Object.entries(MA_LABELS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        )}
        {RULE_TYPES_WITH_THRESHOLD.includes(newType) && (
          <input
            type="number"
            value={newThreshold}
            onChange={(event) => setNewThreshold(Number(event.target.value))}
            className="alerts-panel__threshold"
            title={RULE_TYPES_WITH_LEVEL_INPUT.includes(newType) ? "Nivel de precio" : "Umbral"}
          />
        )}
        <button type="button" className="secondary-button" onClick={submitForm}>
          {editingId ? "Guardar" : "Agregar"}
        </button>
        {editingId && (
          <button type="button" className="secondary-button" onClick={resetForm}>
            Cancelar
          </button>
        )}
      </div>

      <ul className="alerts-panel__list">
        {sortedRules.length === 0 && <li className="alerts-panel__empty">Sin alertas creadas todavía.</li>}
        {sortedRules.map((rule) => {
          const isLive = rule.asset === asset && rule.interval === interval;
          const isTriggered = triggeredIds.has(rule.id);
          return (
            <li
              key={rule.id}
              className={`alerts-panel__rule${isTriggered ? " alerts-panel__rule--triggered" : ""}${
                !rule.enabled ? " alerts-panel__rule--disabled" : ""
              }`}
            >
              <input
                type="checkbox"
                checked={rule.enabled}
                onChange={() => toggleEnabled(rule.id)}
                title={rule.enabled ? "Desactivar" : "Activar"}
              />
              <span className="alerts-panel__rule-text">
                {describeRule(rule)}
                {isLive && <span className="alerts-panel__live-badge">en vivo</span>}
              </span>
              <span className="alerts-panel__rule-actions">
                <button type="button" className="icon-button" onClick={() => startEdit(rule)} aria-label="Editar alerta">
                  ✎
                </button>
                <button type="button" className="icon-button" onClick={() => removeRule(rule.id)} aria-label="Eliminar alerta">
                  ✕
                </button>
              </span>
            </li>
          );
        })}
      </ul>

      <div className="alerts-panel__history">
        <div className="panel-subtitle-row">
          <h4 className="panel-subtitle">Historial</h4>
          {history.length > 0 && (
            <button type="button" className="icon-button" onClick={() => setHistory([])}>
              Limpiar
            </button>
          )}
        </div>
        <ul className="alerts-panel__list alerts-panel__history-list">
          {history.length === 0 && <li className="alerts-panel__empty">Todavía no se disparó ninguna alerta.</li>}
          {history.map((entry) => (
            <li key={entry.id} className="alerts-panel__history-item">
              <span className="alerts-panel__history-time">{formatTimestamp(entry.timestamp)}</span>
              <span>{entry.message}</span>
            </li>
          ))}
        </ul>
      </div>

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
