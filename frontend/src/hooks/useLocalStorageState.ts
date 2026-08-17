/**
 * Estado de React persistido en `localStorage` (Fase 8d) — usado por
 * `AlertsPanel` (reglas de alerta) y `DrawingTools` (dibujos sobre el
 * gráfico). Cada consumidor usa una clave FIJA durante todo el ciclo de
 * vida del componente (nunca cambia dinámicamente con el activo/timeframe
 * seleccionado) y filtra internamente por activo/timeframe si hace falta
 * — así se evita el caso raro de sincronizar el hook con una clave que
 * cambia en caliente.
 */

import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

export function useLocalStorageState<T>(key: string, initialValue: T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      return raw !== null ? (JSON.parse(raw) as T) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // localStorage puede fallar (modo privado, cuota excedida) — no es
      // crítico, el estado sigue funcionando en memoria durante la sesión.
    }
  }, [key, value]);

  return [value, setValue];
}
