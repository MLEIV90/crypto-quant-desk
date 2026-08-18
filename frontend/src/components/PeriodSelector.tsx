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
 */

export type PeriodKey = "1W" | "1M" | "3M" | "6M" | "1A" | "3A" | "todo";

export const DEFAULT_PERIOD: PeriodKey = "3M";

const PERIOD_OPTIONS: { key: PeriodKey; label: string }[] = [
  { key: "1W", label: "1W" },
  { key: "1M", label: "1M" },
  { key: "3M", label: "3M" },
  { key: "6M", label: "6M" },
  { key: "1A", label: "1A" },
  { key: "3A", label: "3A" },
  { key: "todo", label: "Todo" },
];

// Velas por período, por intervalo. "1W"/"todo" usan los números pedidos
// explícitamente (7/168 y 3000/57000, cubriendo desde 2018 en diario y
// desde 2020 en horario — ver scripts/export_snapshot.py::DEFAULT_START_BY_INTERVAL).
// El resto sigue la misma aproximación calendario (días * 24 = horas).
const CANDLES_PER_PERIOD: Record<PeriodKey, { "1d": number; "1h": number }> = {
  "1W": { "1d": 7, "1h": 168 },
  "1M": { "1d": 30, "1h": 720 },
  "3M": { "1d": 90, "1h": 2160 },
  "6M": { "1d": 180, "1h": 4320 },
  "1A": { "1d": 365, "1h": 8760 },
  "3A": { "1d": 1095, "1h": 26280 },
  todo: { "1d": 3000, "1h": 57000 },
};

// `/api/ohlcv` y `/api/studies` acotan `limit` a 5000 (`le=5000` en
// `api/main.py`, backend intacto en esta fase) — "Todo" en horario pediría
// 57000, que la API rechazaría con 422. Se acota acá al máximo real que el
// backend acepta, en vez de dejar que el botón "Todo" falle en horario.
const MAX_API_LIMIT = 5000;

export function candleLimitForPeriod(period: PeriodKey, interval: string): number {
  const table = CANDLES_PER_PERIOD[period];
  const raw = interval === "1h" ? table["1h"] : table["1d"];
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
    </div>
  );
}
