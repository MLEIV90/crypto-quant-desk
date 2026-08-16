/**
 * Selector de activo + timeframe, alimentado por `/api/assets` (Fase 8a) —
 * solo dispara los callbacks que le pasa `App.tsx`, no guarda estado
 * propio ni llama a la API directamente.
 *
 * `showTimeframe` (Fase 8c): el timeframe solo tiene sentido en la vista
 * "Análisis Técnico" — el resto de los endpoints (`/api/risk`,
 * `/api/backtest`, `/api/prediction`) son siempre diarios (ver
 * `api/main.py::DEFAULT_RISK_INTERVAL`), así que mostrar el selector ahí
 * sugeriría, de forma engañosa, que cambia algo que en realidad no afecta.
 */

const TIMEFRAME_LABELS: Record<string, string> = {
  "1d": "Diario (1d)",
  "1h": "Horario (1h)",
};

interface AssetSelectorProps {
  assets: string[];
  timeframes: string[];
  asset: string;
  timeframe: string;
  onAssetChange: (asset: string) => void;
  onTimeframeChange: (timeframe: string) => void;
  showTimeframe?: boolean;
}

export function AssetSelector({
  assets,
  timeframes,
  asset,
  timeframe,
  onAssetChange,
  onTimeframeChange,
  showTimeframe = true,
}: AssetSelectorProps) {
  return (
    <div className="asset-selector">
      <label className="asset-selector__field">
        Activo
        <select value={asset} onChange={(event) => onAssetChange(event.target.value)}>
          {assets.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>
      {showTimeframe && (
        <label className="asset-selector__field">
          Timeframe
          <select value={timeframe} onChange={(event) => onTimeframeChange(event.target.value)}>
            {timeframes.map((item) => (
              <option key={item} value={item}>
                {TIMEFRAME_LABELS[item] ?? item}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
