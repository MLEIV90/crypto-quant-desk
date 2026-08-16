/**
 * Paleta del gráfico — mismos colores (hex) que `app/theme.py` de la
 * terminal de escritorio (PySide6), para que el frontend web y la app de
 * escritorio se vean como el mismo producto. El pulido visual completo
 * (fondo/layout de toda la página) es Fase 8c; esto es solo lo que
 * necesita `components/Chart.tsx` para dibujar.
 */

export const COLORS = {
  background: "#0f172a",
  panelBackground: "#1e293b",
  border: "#334155",
  text: "#e2e8f0",
  textMuted: "#94a3b8",
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
} as const;
