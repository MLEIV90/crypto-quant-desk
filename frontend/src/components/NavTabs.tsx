/**
 * Navegación entre vistas (Fase 8c) — puramente presentacional, el estado
 * de qué vista está activa vive en `App.tsx`.
 */

export type ViewKey =
  | "tecnico"
  | "riesgo"
  | "backtest"
  | "research"
  | "estadistica"
  | "comparacion"
  | "arbitraje"
  | "correlacion";

const VIEWS: { key: ViewKey; label: string }[] = [
  { key: "tecnico", label: "Análisis Técnico" },
  { key: "riesgo", label: "Riesgo" },
  { key: "backtest", label: "Backtest" },
  { key: "research", label: "Research" },
  { key: "estadistica", label: "Estadística" },
  { key: "comparacion", label: "Comparación" },
  { key: "arbitraje", label: "Arbitraje" },
  { key: "correlacion", label: "Correlación" },
];

interface NavTabsProps {
  active: ViewKey;
  onChange: (view: ViewKey) => void;
}

export function NavTabs({ active, onChange }: NavTabsProps) {
  return (
    <nav className="nav-tabs" aria-label="Navegación principal">
      {VIEWS.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className={`nav-tabs__item${active === key ? " nav-tabs__item--active" : ""}`}
          onClick={() => onChange(key)}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}
