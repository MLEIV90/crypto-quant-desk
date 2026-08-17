/**
 * Barra de estado de datos (Fase 9a) — SIEMPRE visible cerca del selector
 * de activo: muestra la última fecha del snapshot local y su antigüedad
 * (`/api/data-status`, liviano, sin red), resaltada en ámbar si está
 * vieja. El botón "Actualizar datos" dispara `/api/refresh` (el ÚNICO
 * endpoint de la API que toca Binance) y, al terminar, invalida las
 * queries de este activo para que todas las vistas se refresquen solas.
 */

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, getDataStatus, postRefresh } from "../api";

interface DataStatusBarProps {
  asset: string;
  interval: string;
}

interface RefreshMessage {
  kind: "success" | "error";
  text: string;
}

function formatFecha(iso: string): string {
  return new Date(iso).toLocaleString();
}

export function DataStatusBar({ asset, interval }: DataStatusBarProps) {
  const queryClient = useQueryClient();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [message, setMessage] = useState<RefreshMessage | null>(null);

  const statusQuery = useQuery({
    queryKey: ["data-status", asset, interval],
    queryFn: () => getDataStatus(asset, interval),
  });
  const status = statusQuery.data;

  async function handleRefresh() {
    setIsRefreshing(true);
    setMessage(null);
    try {
      const result = await postRefresh(asset, interval);
      const fecha = formatFecha(result.ultima_fecha);
      setMessage({
        kind: "success",
        text: result.ya_actualizado
          ? `Ya estaba al día — datos al ${fecha}.`
          : `Agregadas ${result.filas_agregadas} vela${result.filas_agregadas === 1 ? "" : "s"} — datos al ${fecha}.`,
      });
      await queryClient.invalidateQueries({ predicate: (query) => query.queryKey.includes(asset) });
    } catch (error) {
      setMessage({ kind: "error", text: error instanceof ApiError ? error.message : String(error) });
    } finally {
      setIsRefreshing(false);
    }
  }

  return (
    <div className="data-status-bar">
      <div className="data-status-bar__info">
        {statusQuery.isLoading && <span className="view-note">Consultando estado de los datos…</span>}
        {statusQuery.isError && (
          <span className="data-status-bar__stale">No se pudo consultar el estado de los datos.</span>
        )}
        {status && (
          <span className={status.desactualizado ? "data-status-bar__stale" : "data-status-bar__fresh"}>
            Datos al {formatFecha(status.ultima_fecha)} — {status.antiguedad_texto}
            {status.desactualizado ? " (desactualizado)" : ""}
          </span>
        )}
      </div>

      <button
        type="button"
        className="secondary-button"
        onClick={() => void handleRefresh()}
        disabled={isRefreshing}
      >
        {isRefreshing && <span className="spinner spinner--inline" />}
        {isRefreshing ? "Actualizando…" : "Actualizar datos"}
      </button>

      {message && (
        <span className={message.kind === "success" ? "data-status-bar__success" : "data-status-bar__error"}>
          {message.text}
        </span>
      )}
    </div>
  );
}
