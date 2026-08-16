/**
 * Selector de activo + timeframe, alimentado por `/api/assets` (Fase 8a) —
 * solo dispara los callbacks que le pasa `App.tsx`, no guarda estado
 * propio ni llama a la API directamente.
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
}

export function AssetSelector({
  assets,
  timeframes,
  asset,
  timeframe,
  onAssetChange,
  onTimeframeChange,
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
    </div>
  );
}
