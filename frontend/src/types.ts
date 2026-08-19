/**
 * Tipos TypeScript que reflejan los esquemas Pydantic de la API
 * (`api/models.py`, Fases 8a/8c) — un tipo por respuesta que consume este
 * frontend.
 *
 * Convención: fechas llegan como string ISO 8601 UTC (p. ej.
 * "2026-08-12T00:00:00Z") — se parsean recién donde hace falta un
 * timestamp numérico (ver `components/Chart.tsx`/`components/LineChartPanel.tsx`),
 * nunca acá.
 */

export interface AssetsResponse {
  activos: string[];
  timeframes: string[];
}

export interface Candle {
  fecha: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

export interface OHLCVResponse {
  asset: string;
  interval: string;
  velas: Candle[];
}

export interface SoporteResistencia {
  resistencia: number[];
  soporte: number[];
  precio_actual: number;
}

export interface StudiesResponse {
  asset: string;
  interval: string;
  fechas: string[];
  sma_20: (number | null)[];
  sma_50: (number | null)[];
  ema_12: (number | null)[];
  ema_26: (number | null)[];
  bb_upper: (number | null)[];
  bb_mid: (number | null)[];
  bb_lower: (number | null)[];
  rsi_14: (number | null)[];
  macd: (number | null)[];
  macd_signal: (number | null)[];
  macd_hist: (number | null)[];
  stoch_k: (number | null)[];
  stoch_d: (number | null)[];
  vwap: (number | null)[];
  obv: (number | null)[];
  ichimoku_tenkan: (number | null)[];
  ichimoku_kijun: (number | null)[];
  ichimoku_senkou_a: (number | null)[];
  ichimoku_senkou_b: (number | null)[];
  ichimoku_chikou: (number | null)[];
  fibonacci: Record<string, number> | null;
  soporte_resistencia: SoporteResistencia;
  pivotes: Record<string, number>;
}

export interface DesempenoHistorico {
  cagr_sugeridor: number;
  sharpe_sugeridor: number;
  max_drawdown_sugeridor: number;
  n_trades_sugeridor: number;
  cagr_buy_and_hold: number;
  sharpe_buy_and_hold: number;
  max_drawdown_buy_and_hold: number;
}

export interface SuggesterResponse {
  sugerencia: "COMPRAR" | "VENDER" | "ESPERAR";
  votos_alcistas: number;
  votos_bajistas: number;
  votos_neutrales: number;
  confianza: number;
  detalle: Record<string, "alcista" | "bajista" | "neutral">;
  desempeno_historico: DesempenoHistorico;
}

export interface RiskResponse {
  asset: string;
  vol_realizada: number;
  modelo_garch: string;
  vol_garch: number;
  regimen: "calma" | "normal" | "tension" | null;
  var95: number;
  es95: number;
  accion: "LONG" | "FLAT" | "SHORT";
  score: number;
  tamano_sugerido: number;
  ultima_fecha: string;
}

export interface EquityPoint {
  fecha: string;
  valor: number;
}

export interface BacktestResponse {
  asset: string;
  metrics_estrategia: Record<string, number>;
  metrics_buy_and_hold: Record<string, number>;
  equity_curve_estrategia: EquityPoint[];
  equity_curve_buy_and_hold: EquityPoint[];
}

export interface GarchSeriesResponse {
  asset: string;
  fechas: string[];
  vol_condicional: (number | null)[];
  modelo_garch: string;
  regimen_actual: "calma" | "normal" | "tension" | null;
}

export interface DataStatusResponse {
  asset: string;
  interval: string;
  ultima_fecha: string;
  antiguedad_segundos: number;
  antiguedad_texto: string;
  desactualizado: boolean;
}

export interface RefreshResponse {
  asset: string;
  interval: string;
  filas_agregadas: number;
  ultima_fecha: string;
  ya_actualizado: boolean;
}

export interface SeasonalityBucket {
  bucket: number;
  retorno_medio: number;
  mediana: number | null;
  desvio: number | null;
  n: number;
}

export interface AutocorrelationPoint {
  lag: number;
  acf_retornos: number | null;
  acf_retornos2: number | null;
}

export interface PeriodogramTopPeriod {
  periodo_dias: number;
  potencia: number;
}

export interface PeriodogramResponse {
  frecuencias: number[];
  potencia: (number | null)[];
  top_periodos: PeriodogramTopPeriod[];
}

export interface AdfResult {
  estadistico: number;
  p_valor: number;
  n_lags: number;
  n_obs: number;
  valores_criticos: Record<string, number>;
  es_estacionaria: boolean;
}

export interface StatsResponse {
  asset: string;
  interval: string;
  estacionalidad_mensual: SeasonalityBucket[];
  estacionalidad_semanal: SeasonalityBucket[];
  estacionalidad_horaria: SeasonalityBucket[] | null;
  autocorrelacion: AutocorrelationPoint[];
  periodograma: PeriodogramResponse;
  adf_precio: AdfResult;
  adf_retornos: AdfResult;
  halvings_btc: string[] | null;
}

export interface CompareResponse {
  assets: string[];
  interval: string;
  fechas: string[];
  series: Record<string, (number | null)[]>;
  rendimiento_total_pct: Record<string, number>;
}

export interface PredictionResponse {
  asset: string;
  used_onchain: boolean;
  onchain_columns: string[];
  ultima_fecha: string;
  prediccion_clase: "LONG" | "FLAT" | "SHORT";
  prediccion_confianza: number;
  prediccion_proba: Record<string, number>;
  accuracy_media: number;
  baseline_azar: number;
  baseline_mayoritaria: number;
  supera_azar: boolean;
  supera_mayoritaria: boolean;
  roc_auc_media: number;
  top_features: [string, number][];
}
