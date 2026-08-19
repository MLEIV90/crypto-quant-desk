/**
 * Selector MÚLTIPLE de activos (Fase 12a, vista "Comparación") — a
 * diferencia de `AssetSelector` (un solo activo "activo" para el resto de
 * la app), acá se puede marcar cualquier subconjunto del universo para
 * compararlos entre sí. Reutiliza las clases `.toggles__group`/
 * `.toggle-chip` ya existentes (mismo look que los checkboxes de
 * overlays/osciladores) en vez de duplicar estilos.
 */

interface AssetMultiSelectProps {
  assets: string[];
  selected: Set<string>;
  onToggle: (asset: string) => void;
}

export function AssetMultiSelect({ assets, selected, onToggle }: AssetMultiSelectProps) {
  return (
    <div className="toggles__group">
      {assets.map((asset) => (
        <label key={asset} className={`toggle-chip${selected.has(asset) ? " toggle-chip--active" : ""}`}>
          <input type="checkbox" checked={selected.has(asset)} onChange={() => onToggle(asset)} />
          {asset}
        </label>
      ))}
    </div>
  );
}
