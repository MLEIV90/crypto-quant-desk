/**
 * Estado de carga/error consistente en TODAS las vistas (Fase 8c) — evita
 * que cada vista invente su propio mensaje/estilo de "cargando"/"error".
 */

import type { ReactNode } from "react";

interface StatusMessageProps {
  kind: "loading" | "error";
  children: ReactNode;
}

export function StatusMessage({ kind, children }: StatusMessageProps) {
  return (
    <div className={`status status--${kind}`} role={kind === "error" ? "alert" : "status"}>
      {kind === "loading" && <span className="spinner" aria-hidden="true" />}
      <span>{children}</span>
    </div>
  );
}
