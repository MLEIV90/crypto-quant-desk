/**
 * Textos de ayuda centralizados (Fase 9b) — un solo lugar para mantener y
 * eventualmente traducir todas las explicaciones que muestra `InfoTooltip`
 * en la app. Ningún componente hardcodea su propio texto de ayuda.
 *
 * CRITERIO DE REDACCIÓN, el mismo que ya rige el resto del proyecto (ver
 * `signals/suggester.py`, `ml/models.py`): para los estudios SIN edge
 * demostrado (que son la mayoría — Fibonacci, soporte/resistencia,
 * pivotes, y en general cualquier indicador técnico clásico), el texto
 * dice explícitamente que no predice y que no tiene respaldo estadístico
 * probado en este proyecto. Nada de lenguaje que sugiera que un estudio
 * "anticipa" o "confirma" un movimiento futuro.
 */

import type { OverlayKey } from "./components/StudyToggles";
import type { OscillatorKey } from "./components/OscillatorPanel";

export const OVERLAY_HELP: Record<OverlayKey, string> = {
  sma20: "Promedio simple del precio de cierre de las últimas 20 velas. Suaviza el ruido de corto plazo para ver hacia dónde viene la tendencia — no anticipa hacia dónde va.",
  sma50: "Igual que la SMA 20 pero con 50 velas: más lenta para reaccionar, menos ruido. Se usa como referencia de tendencia de más largo plazo.",
  ema12: "Promedio exponencial de 12 velas: a diferencia de la SMA, pondera más los precios recientes. Es una de las dos medias que arman el MACD.",
  ema26: "Promedio exponencial de 26 velas. Junto con la EMA 12 forma la línea del MACD (ver el oscilador MACD).",
  bollinger: "Banda de ±2 desvíos estándar alrededor de una media móvil. Cuando se abre hubo más volatilidad reciente; cuando se cierra, menos. No indica en qué dirección va a romper el precio.",
  fibonacci: "Niveles de retroceso/extensión populares entre traders técnicos, calculados sobre un máximo y un mínimo reciente. SIN respaldo estadístico probado en este proyecto — es una referencia visual, no una regla.",
  supportResistance: "Zonas de precio donde el activo rebotó o se frenó recientemente. Que haya pasado antes no garantiza que vuelva a pasar — es una referencia visual, no una predicción.",
  pivots: "Niveles calculados a partir del rango (alto/bajo/cierre) de la vela anterior. Muy usados en trading de corto plazo; el precio puede perforarlos sin aviso.",
};

export const OSCILLATOR_HELP: Record<OscillatorKey, string> = {
  rsi: "Mide si el activo está sobrecomprado (>70) o sobrevendido (<30) según la velocidad y magnitud de los cambios de precio recientes. No predice: muestra momentum reciente, nada más.",
  macd: "Diferencia entre la EMA 12 y la EMA 26 del precio, con su propia media de señal. Un cruce entre la línea MACD y su señal se lee como cambio de momentum — no es una señal de compra/venta garantizada.",
  stochastic: "Compara el cierre actual contra el rango de precios reciente, en una escala de 0 a 100. Por encima de 80 sugiere sobrecompra, por debajo de 20 sobreventa — mide momentum, igual que el RSI, no anticipa el precio.",
};

export const RISK_INTRO_HELP =
  "Estas son medidas de RIESGO calculadas sobre la historia del activo — ninguna predice el precio de mañana. " +
  "Sirven para decidir CUÁNTO exponerte (tamaño de posición), no CUÁNDO vas a acertar.";

export const RISK_METRIC_HELP = {
  volRealizada:
    "Qué tan movido estuvo el precio en el pasado reciente (volatilidad histórica, anualizada). Más alta = " +
    "movimientos diarios más grandes, en ambas direcciones.",
  modeloGarch:
    "El modelo estadístico (familia GARCH) que mejor ajustó la volatilidad histórica de este activo, elegido " +
    "automáticamente por criterio AIC — no lo elige un humano a mano.",
  volGarch:
    "Estimación de volatilidad de ese modelo GARCH: le da más peso a los movimientos recientes que la volatilidad " +
    "realizada simple. Es una proyección de corto plazo sobre datos pasados, no una certeza sobre el futuro.",
  regimen:
    "CALMA = volatilidad baja respecto a la propia historia del activo. NORMAL = dentro de lo típico. TENSIÓN = " +
    "volatilidad alta. Útil para ajustar el tamaño de posición, no para adivinar el momento de entrar o salir.",
  var95:
    "En un día malo típico (el peor 5% de los casos históricos), podrías perder alrededor de esto. Es una medida " +
    "de riesgo estadística sobre el pasado, no una predicción de lo que va a pasar mañana.",
  es95:
    "Si ese día malo (el peor 5%) efectivamente ocurre, esto es la pérdida PROMEDIO esperada en ese escenario — " +
    "siempre es igual o mayor al VaR, porque mira más adentro de la cola mala de la distribución.",
  senal:
    "Dirección sugerida por el motor de señales (tendencia + momentum + reversión a la media combinados en un " +
    "score). Es un dato de apoyo, no una recomendación de inversión.",
  sizing:
    "Tamaño de posición sugerido por 'vol targeting': ajusta la exposición para apuntar a una volatilidad objetivo " +
    "del book, reduciendo el tamaño cuando el activo está más volátil de lo normal.",
};

export const SUGGESTER_HELP =
  "Es un VOTO de varios estudios técnicos (RSI, medias, MACD, estocástico, Bollinger, pivote) combinados por " +
  "mayoría — no un modelo entrenado ni una IA que aprendió patrones. Antes de hacerle caso, mirá su desempeño " +
  "histórico (Sharpe/CAGR/drawdown) de acá abajo: si no le gana con claridad al buy & hold, tratalo como un dato " +
  "más entre varios, no como una orden de operar.";

export const BACKTEST_EQUITY_HELP =
  "Evolución de $1 invertido desde el inicio del período, para la estrategia del engine y para comprar-y-mantener " +
  "en paralelo. Muestra CÓMO se llegó al resultado final (con qué subidas y caídas en el camino), no solo el " +
  "número final.";

export const BACKTEST_METRIC_HELP: Record<string, string> = {
  cagr: "Crecimiento anual compuesto: a qué tasa anual equivalente creció el capital en promedio durante el período.",
  sharpe: "Retorno ajustado por riesgo: cuánto retorno se obtuvo por unidad de volatilidad asumida. Más alto es mejor; no distingue una caída brusca de una subida brusca.",
  sortino: "Como el Sharpe, pero solo penaliza la volatilidad A LA BAJA — las subidas bruscas no lo perjudican.",
  max_drawdown: "La peor caída, de punta a punta, desde un máximo hasta el mínimo posterior en todo el período — cuánto tuviste que aguantar en el peor momento.",
  calmar: "Retorno anualizado dividido por el máximo drawdown: cuánto retorno obtuviste por cada unidad de 'dolor' de la peor caída.",
  n_trades: "Cantidad de operaciones (cambios de posición) que ejecutó la estrategia durante el período.",
  turnover_total: "Cuánto 'movimiento' de posición hubo en total — más turnover implica más costos de transacción reales.",
};
