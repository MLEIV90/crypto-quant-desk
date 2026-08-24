/**
 * Tema centralizado (Fase 8c): ÚNICA fuente de verdad de la paleta de
 * colores de todo el frontend — mismos valores que `app/theme.py` de la
 * terminal de escritorio (PySide6), para que ambos frontends se sientan
 * el mismo producto.
 *
 * `COLORS` se consume de dos formas:
 * 1. Directo en TypeScript, donde lightweight-charts necesita un string
 *    hex literal (`components/Chart.tsx`, `components/LineChartPanel.tsx`).
 * 2. Como variables CSS (`--color-*`), inyectadas UNA vez al arrancar la
 *    app (`applyThemeCssVariables`, llamado desde `main.tsx`) — el resto
 *    del CSS (`index.css`, `App.css`) las consume con `var(--color-...)`
 *    en vez de repetir valores hex a mano, para que no puedan divergir.
 */

export const COLORS = {
  background: "#0f172a",
  panelBackground: "#1e293b",
  panelBackgroundElevated: "#243046",
  border: "#334155",
  borderStrong: "#475569",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
  textFaint: "#64748b",
  accent: "#38bdf8",
  accentMuted: "#0ea5e9",
  success: "#22c55e",
  danger: "#ef4444",
  warning: "#eab308",
  candleUp: "#22c55e",
  candleDown: "#ef4444",
  sma20: "#f59e0b",
  sma50: "#eab308",
  ema12: "#38bdf8",
  ema26: "#818cf8",
  bollinger: "#a78bfa",
  fibonacci: "#fbbf24",
  support: "#22c55e",
  resistance: "#ef4444",
  pivot: "#e2e8f0",
  rsi: "#38bdf8",
  macdLine: "#38bdf8",
  macdSignal: "#f59e0b",
  stochK: "#38bdf8",
  stochD: "#f59e0b",
  equityStrategy: "#38bdf8",
  equityBuyHold: "#f59e0b",
  vwap: "#2dd4bf",
  obv: "#2dd4bf",
  ichimokuTenkan: "#f97316",
  ichimokuKijun: "#3b82f6",
  ichimokuSenkouA: "#22c55e",
  ichimokuSenkouB: "#ef4444",
  ichimokuChikou: "#94a3b8",
  volumeProfileBar: "#38bdf8",
  volumeProfilePoc: "#ec4899",
} as const;

export type ColorKey = keyof typeof COLORS;

/** Sugerencia/dirección -> color: COMPRAR/LONG = verde, ESPERAR/FLAT =
 * gris, VENDER/SHORT = rojo — un solo mapeo para el sugeridor (Fase 7a) y
 * el panel de riesgo (`accion` de `signals.engine`), que usan vocabularios
 * distintos para el mismo concepto.
 */
export const DIRECTION_COLORS: Record<string, string> = {
  COMPRAR: COLORS.success,
  LONG: COLORS.success,
  ESPERAR: COLORS.textMuted,
  FLAT: COLORS.textMuted,
  VENDER: COLORS.danger,
  SHORT: COLORS.danger,
};

/** Régimen de volatilidad (`models.garch.volatility_regime`) -> color. */
export const REGIME_COLORS: Record<string, string> = {
  calma: COLORS.success,
  normal: COLORS.warning,
  tension: COLORS.danger,
};

/** Régimen -> color para la FRANJA histórica de `RegimeStrip` (Fase 20a) —
 * gris para "normal" en vez del ámbar de `REGIME_COLORS`: en una franja
 * continua que cubre TODA la historia, "normal" es el estado por defecto
 * la mayor parte del tiempo, así que pintarlo con un color de alerta
 * (ámbar) dominaría visualmente la franja y opacaría los tramos de calma/
 * tensión, que son los que de verdad importa distinguir de un vistazo.
 * `REGIME_COLORS` sigue igual para la tarjeta de "régimen actual" (un
 * único punto en el tiempo, donde ámbar como color intermedio sí tiene
 * sentido).
 */
export const REGIME_STRIP_COLORS: Record<string, string> = {
  calma: COLORS.success,
  normal: COLORS.textFaint,
  tension: COLORS.danger,
};

/** Voto de un estudio del sugeridor (`signals.suggester`) -> color. */
export const VOTE_COLORS: Record<string, string> = {
  alcista: COLORS.success,
  bajista: COLORS.danger,
  neutral: COLORS.textMuted,
};

/** Activo (`config.UNIVERSE`) -> color, usando el color de marca de cada
 * moneda (Fase 12a, vista "Comparación") — más reconocible a simple vista
 * que asignar colores genéricos por índice. `FALLBACK_ASSET_COLORS` cubre
 * cualquier ticker fuera de esta lista (no debería pasar con el universo
 * actual, pero evita quedarse sin color si se agrega un activo nuevo).
 */
export const ASSET_COLORS: Record<string, string> = {
  BTC: "#f7931a",
  ETH: "#627eea",
  SOL: "#9945ff",
  BNB: "#f0b90b",
  LTC: "#345d9d",
};

export const FALLBACK_ASSET_COLORS: string[] = [
  COLORS.accent, COLORS.success, COLORS.warning, COLORS.danger, COLORS.bollinger,
];

export function colorForAsset(asset: string, index: number): string {
  return ASSET_COLORS[asset] ?? FALLBACK_ASSET_COLORS[index % FALLBACK_ASSET_COLORS.length];
}

function toKebabCase(camelCase: string): string {
  return camelCase.replace(/([A-Z])/g, "-$1").toLowerCase();
}

/** Inyecta `COLORS` como variables CSS (`--color-background`,
 * `--color-panel-background`, etc.) en `:root` — se llama UNA vez al
 * arrancar la app (`main.tsx`), antes de montar `<App />`, para que el
 * CSS ya las tenga disponibles en el primer render.
 */
export function applyThemeCssVariables(): void {
  const root = document.documentElement;
  for (const [key, value] of Object.entries(COLORS)) {
    root.style.setProperty(`--color-${toKebabCase(key)}`, value);
  }
}
