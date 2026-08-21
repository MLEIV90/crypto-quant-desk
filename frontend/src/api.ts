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

import type {
  AssetsResponse,
  BacktestResponse,
  CompareResponse,
  CorrelationResponse,
  DataStatusResponse,
  GarchSeriesResponse,
  OHLCVResponse,
  PairDetailResponse,
  PairScreeningResponse,
  PredictionResponse,
  RefreshResponse,
  RiskResponse,
  StatsResponse,
  StudiesResponse,
  SuggesterResponse,
  VolumeProfileResponse,
} from "./types";

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

async function apiPost<T>(path: string, params: Record<string, string | number> = {}): Promise<T> {
  const url = new URL(path, API_BASE_URL);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, String(value));
  }

  let response: Response;
  try {
    response = await fetch(url.toString(), { method: "POST" });
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

/**
 * Volume Profile (Fase 13a): volumen operado por nivel de precio (POC +
 * Value Area) — ver `api/main.py::get_volume_profile`. Mismo `limit` que
 * `/api/ohlcv`/`/api/studies`, para que el perfil respete el período
 * elegido en `PeriodSelector` en vez de recalcularse sobre todo el histórico.
 */
export function getVolumeProfile(
  asset: string,
  interval: string,
  limit: number,
): Promise<VolumeProfileResponse> {
  return apiGet<VolumeProfileResponse>("/api/volume-profile", { asset, interval, limit });
}

export function getSuggester(asset: string, interval: string): Promise<SuggesterResponse> {
  return apiGet<SuggesterResponse>("/api/suggester", { asset, interval });
}

/** Lento (ajusta un modelo GARCH) — ver `api/main.py`. */
export function getRisk(asset: string): Promise<RiskResponse> {
  return apiGet<RiskResponse>("/api/risk", { asset });
}

export function getBacktest(asset: string): Promise<BacktestResponse> {
  return apiGet<BacktestResponse>("/api/backtest", { asset });
}

/** Lento (ajusta un modelo GARCH) — ver `api/main.py`. */
export function getGarchSeries(asset: string): Promise<GarchSeriesResponse> {
  return apiGet<GarchSeriesResponse>("/api/garch-series", { asset });
}

/**
 * MUY LENTO (15-30s: entrena XGBoost con validación purgeada, ver
 * `api/main.py`) — SIEMPRE disparar esto ante una acción explícita del
 * usuario (un botón), nunca automáticamente al montar una vista.
 */
export function getPrediction(asset: string): Promise<PredictionResponse> {
  return apiGet<PredictionResponse>("/api/prediction", { asset });
}

/** Solo lee el snapshot local (rápido, sin red) — ver `api/main.py`. */
export function getDataStatus(asset: string, interval: string): Promise<DataStatusResponse> {
  return apiGet<DataStatusResponse>("/api/data-status", { asset, interval });
}

/**
 * ÚNICO endpoint de toda la API que toca la red (Binance) — puede tardar
 * varios segundos. Dispararlo solo ante una acción explícita del usuario
 * (el botón "Actualizar datos"), nunca automáticamente.
 */
export function postRefresh(asset: string, interval: string): Promise<RefreshResponse> {
  return apiPost<RefreshResponse>("/api/refresh", { asset, interval });
}

/**
 * Estacionalidad, autocorrelación, ciclos (periodograma) y estacionariedad
 * (ADF) — ver `api/main.py::get_stats` (Fase 11). Liviano (sin ajuste de
 * modelos), pero recalcula estadística sobre TODO el histórico disponible
 * cada vez, no solo el rango visible del gráfico.
 */
export function getStats(asset: string, interval: string): Promise<StatsResponse> {
  return apiGet<StatsResponse>("/api/stats", { asset, interval });
}

/**
 * Comparación de rendimiento normalizado a base 100 entre varios activos —
 * ver `api/main.py::get_compare` (Fase 12a). `assets` va separado por
 * coma (ej. "BTC,ETH,SOL"), igual que espera el endpoint.
 */
export function getCompare(assets: string, interval: string, limit: number): Promise<CompareResponse> {
  return apiGet<CompareResponse>("/api/compare", { assets, interval, limit });
}

/**
 * Matriz de correlación entre activos sobre RETORNOS (Fase 13b) — ver
 * `api/main.py::get_correlation`. `method` es "pearson" o "spearman".
 */
export function getCorrelation(interval: string, limit: number, method: string): Promise<CorrelationResponse> {
  return apiGet<CorrelationResponse>("/api/correlation", { interval, limit, method });
}

/**
 * Ranking honesto de operabilidad de pares — ver `api/main.py::get_pairs_screening`
 * (Fase 12b, `pairs.stability.screen_pairs_stability`). Solo soporta
 * `interval="1d"` (la función reutilizada siempre carga precios diarios).
 */
export function getPairsScreening(interval: string = "1d"): Promise<PairScreeningResponse> {
  return apiGet<PairScreeningResponse>("/api/pairs/screening", { interval });
}

/**
 * Detalle de un par: cointegración, spread, z-score y estabilidad rolling —
 * ver `api/main.py::get_pairs_detail` (Fase 12b). A diferencia del
 * screening, sí respeta `interval`.
 */
export function getPairsDetail(assetY: string, assetX: string, interval: string): Promise<PairDetailResponse> {
  return apiGet<PairDetailResponse>("/api/pairs/detail", { asset_y: assetY, asset_x: assetX, interval });
}

/**
 * Informe PDF descargable (Fase 16b) — ver `api/main.py::get_report`
 * (`reports/pdf_report.py`). MUY LENTO (ajusta un GARCH y corre
 * cointegración rolling sobre todos los pares): del orden de 30-90
 * segundos. A diferencia del resto de las funciones de este archivo, la
 * respuesta no es JSON sino el archivo PDF crudo — se devuelve como `Blob`
 * para que el llamador dispare la descarga en el navegador.
 */
export async function getPdfReport(asset: string, interval: string): Promise<Blob> {
  const url = new URL("/api/report", API_BASE_URL);
  url.searchParams.set("asset", asset);
  url.searchParams.set("interval", interval);

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
    throw new ApiError(body?.detail ?? `Error ${response.status} al llamar /api/report`, response.status);
  }

  return response.blob();
}
