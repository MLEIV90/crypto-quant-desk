/**
 * Selector de período histórico (Fase 10a) — botones tipo pill que
 * reemplazan el `CANDLE_LIMIT` estático que tenían `TechnicalAnalysisView`
 * y `RiskView` (300 velas fijas, ~10 meses en diario). Cada período se
 * traduce a una cantidad de velas SEGÚN el intervalo (1d vs 1h) vía
 * `candleLimitForPeriod` — mismo número de período, distinta cantidad de
 * velas si el usuario está mirando diario u horario.
 *
 * El componente solo expone el estado (`active`/`onChange`); quien lo usa
 * decide qué hacer con el período (armar el `limit` que manda a
 * `/api/ohlcv`/`/api/studies`). Pasar el MISMO `period`/`candleLimit` a
 * ambos endpoints es lo que mantiene sincronizados el precio y los
 * osciladores (RSI/MACD/Estocástico) — ver `views/TechnicalAnalysisView.tsx`.
 *
 * Fase 17b: agrega "4h" y "1w" (derivados por resampleo, ver
 * `data.loaders.RESAMPLED_INTERVALS`) a `CANDLE_DURATION_DAYS` — la
 * cantidad de velas por período ya no es una tabla a mano por intervalo
 * (que había que ampliar cada vez que se agrega uno nuevo), se calcula
 * dividiendo "días calendario del período" por "duración de una vela en
 * días", genérico para cualquier intervalo presente en ese mapa.
 */

import { InfoTooltip } from "./InfoTooltip";
import { PERIOD_SELECTOR_HELP } from "../helpTexts";

export type PeriodKey = "1W" | "1M" | "3M" | "6M" | "1A" | "3A" | "todo";

// Fase 31 (hallazgo A3-01): antes "3M" — las vistas de análisis arrancaban
// mostrando solo 3 meses, ocultando en silencio la mayoría del histórico
// disponible (para BTC, hasta 16 años con la serie larga fusionada de
// CoinMetrics+Binance, ver `data/loaders.py::MERGE_CUTOFF_DATE`) sin que el
// usuario lo notara a menos que supiera tocar el selector. Se cambia el
// default COMPARTIDO a "todo" para las 4 vistas que lo usan (Análisis
// Técnico, Comparación, Arbitraje, Correlación): mostrar de más nunca
// esconde información, mostrar de menos sí — y el usuario siempre puede
// acotar el período a mano si prefiere una ventana más corta.
export const DEFAULT_PERIOD: PeriodKey = "todo";

// Fase 33 (PARTE B, verificación de rendimiento): "todo" es liviano en el
// intervalo DIARIO, que es el que arranca por defecto en `App.tsx`
// (`DEFAULT_INTERVAL = "1d"`) — medido contra el snapshot real: BTC (el
// más largo, con la serie fusionada de Fase 31) son ~5.885 velas / ~750KB
// en `/api/ohlcv` y ~1.05MB en `/api/studies`; el resto del universo son
// menos (2.200-3.200 velas). Nada de eso satura el navegador ni tarda
// perceptiblemente. La combinación realmente pesada es "todo" + intervalo
// HORARIO (~58.000 velas, ~6.7MB en `/api/ohlcv` y ~19.6MB en
// `/api/studies`) — pero esa combinación NO es la que arranca por
// defecto (el usuario tiene que cambiar el timeframe a mano) y ya
// preexistía antes de esta fase; `../downsample.ts` + `Chart.tsx` la
// mitigan del lado del RENDERIZADO (nunca dibuja más de ~6.000 puntos),
// aunque no reduce el peso de la descarga en sí. Por eso "todo" se dejó
// como default en las 4 vistas que lo comparten (Técnico, Comparación,
// Arbitraje, Correlación) sin necesidad de un default más chico para
// ninguna: el caso que de verdad pesa no es el que se activa solo.

const PERIOD_OPTIONS: { key: PeriodKey; label: string }[] = [
  { key: "1W", label: "1W" },
  { key: "1M", label: "1M" },
  { key: "3M", label: "3M" },
  { key: "6M", label: "6M" },
  { key: "1A", label: "1A" },
  { key: "3A", label: "3A" },
  { key: "todo", label: "Todo" },
];

// `/api/ohlcv`/`/api/studies` acotan `limit` a MAX_CANDLE_LIMIT=60_000
// (api/main.py, Fase 11) — bien por encima del histórico real más largo
// (~58.000 velas horarias desde 2020, ~3.150 diarias desde 2018).
const MAX_API_LIMIT = 60_000;

// Días calendario que cubre cada período, salvo "todo" (Fase 11, fix del
// rango "Todo"): en vez de un número inventado que podía quedarse corto
// contra el histórico real, pide directamente el tope que acepta el
// backend — como el backend recorta con `.tail(limit)`, pedir MÁS de lo
// que existe simplemente devuelve TODO lo disponible.
const CALENDAR_DAYS_PER_PERIOD: Record<Exclude<PeriodKey, "todo">, number> = {
  "1W": 7,
  "1M": 30,
  "3M": 90,
  "6M": 180,
  "1A": 365,
  "3A": 1095,
};

// Duración de una vela de cada intervalo, en DÍAS calendario (Fase 17b) —
// única tabla que hay que ampliar al agregar un intervalo nuevo, en vez de
// una fila por cada combinación período x intervalo.
const CANDLE_DURATION_DAYS: Record<string, number> = {
  "1h": 1 / 24,
  "4h": 4 / 24,
  "1d": 1,
  "1w": 7,
};

export function candleLimitForPeriod(period: PeriodKey, interval: string): number {
  if (period === "todo") return MAX_API_LIMIT;
  const days = CALENDAR_DAYS_PER_PERIOD[period];
  const candleDurationDays = CANDLE_DURATION_DAYS[interval] ?? 1;
  const raw = Math.ceil(days / candleDurationDays);
  return Math.min(raw, MAX_API_LIMIT);
}

interface PeriodSelectorProps {
  active: PeriodKey;
  onChange: (period: PeriodKey) => void;
}

export function PeriodSelector({ active, onChange }: PeriodSelectorProps) {
  return (
    <div className="period-selector" role="group" aria-label="Período histórico">
      {PERIOD_OPTIONS.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className={`period-selector__pill${active === key ? " period-selector__pill--active" : ""}`}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
      <InfoTooltip text={PERIOD_SELECTOR_HELP} placement="bottom" />
    </div>
  );
}
