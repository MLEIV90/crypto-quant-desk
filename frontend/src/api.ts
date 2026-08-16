/**
 * Cliente HTTP de la API REST del backend (Fase 8a, `api/main.py`) — capa
 * fina: cada función acá hace un `fetch` a un endpoint y devuelve el JSON
 * ya tipado, sin transformar ni calcular nada (esa regla es la misma que
 * sigue la API del lado del backend: ver `api/main.py`).
 *
 * URL base configurable vía `VITE_API_BASE_URL` (ver `.env.example` y el
 * README de este proyecto) — por defecto apunta a
 * `http://127.0.0.1:8000`, donde corre `uvicorn api.main:app --reload` en
 * desarrollo.
 */

import type { AssetsResponse, OHLCVResponse, StudiesResponse } from "./types";

export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

/** Error de la API con el código HTTP y el detalle que devolvió el backend
 * (`{"detail": "..."}`, la convención de error de FastAPI/HTTPException).
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function apiGet<T>(path: string, params: Record<string, string | number> = {}): Promise<T> {
  const url = new URL(path, API_BASE_URL);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, String(value));
  }

  let response: Response;
  try {
    response = await fetch(url.toString());
  } catch {
    throw new ApiError(
      `No se pudo conectar con la API en ${API_BASE_URL}. ¿Está corriendo "uvicorn api.main:app --reload"?`,
      0,
    );
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new ApiError(body?.detail ?? `Error ${response.status} al llamar ${path}`, response.status);
  }

  return (await response.json()) as T;
}

export function getAssets(): Promise<AssetsResponse> {
  return apiGet<AssetsResponse>("/api/assets");
}

export function getOhlcv(asset: string, interval: string, limit: number): Promise<OHLCVResponse> {
  return apiGet<OHLCVResponse>("/api/ohlcv", { asset, interval, limit });
}

export function getStudies(asset: string, interval: string, limit: number): Promise<StudiesResponse> {
  return apiGet<StudiesResponse>("/api/studies", { asset, interval, limit });
}
