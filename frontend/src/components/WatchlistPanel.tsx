/**
 * Panel de watchlist (Fase 10a) — franja horizontal compacta con las
 * monedas de `config.UNIVERSE`, siempre visible (no depende de la vista
 * activa), para cambiar de activo con un click en vez de tener que usar el
 * selector del header. Reutiliza `/api/data-status` (Fase 9a, para la
 * antigüedad) y `/api/ohlcv` con `limit=2` (para el cambio % contra la
 * vela anterior) — ningún endpoint nuevo.
 *
 * Siempre en diario ("1d"): el cambio % que muestra es el que un usuario
 * espera de un ticker de mercado ("cambio 24h"), independiente del
 * timeframe que tenga seleccionado la vista de Análisis Técnico.
 *
 * Cada entrada es UNA query combinada (data-status + ohlcv en paralelo)
 * con clave `["watchlist", asset]` — el mismo `asset` que usa el resto de
 * las queries de la vista activa, así que el `invalidateQueries` que ya
 * dispara `DataStatusBar` al terminar un refresh (predicate sobre
 * `queryKey.includes(asset)`) también refresca la entrada del watchlist
 * del activo recién actualizado, sin código extra.
 */

import { useQueries } from "@tanstack/react-query";
import { getDataStatus, getOhlcv } from "../api";
import { InfoTooltip } from "./InfoTooltip";
import { WATCHLIST_HELP } from "../helpTexts";

const WATCHLIST_INTERVAL = "1d";
const CHANGE_CANDLE_LIMIT = 2;

interface WatchlistEntry {
  price: number | null;
  changePct: number | null;
  desactualizado: boolean;
}

async function fetchWatchlistEntry(asset: string): Promise<WatchlistEntry> {
  const [status, ohlcv] = await Promise.all([
    getDataStatus(asset, WATCHLIST_INTERVAL),
    getOhlcv(asset, WATCHLIST_INTERVAL, CHANGE_CANDLE_LIMIT),
  ]);

  const velas = ohlcv.velas;
  const last = velas[velas.length - 1];
  const prev = velas[velas.length - 2];
  const price = last ? last.close : null;
  const changePct = last && prev && prev.close !== 0 ? (last.close - prev.close) / prev.close : null;

  return { price, changePct, desactualizado: status.desactualizado };
}

function formatPrice(price: number): string {
  return `$${price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatChangePct(pct: number): string {
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${(pct * 100).toFixed(2)}%`;
}

interface WatchlistPanelProps {
  assets: string[];
  activeAsset: string;
  onSelectAsset: (asset: string) => void;
}

export function WatchlistPanel({ assets, activeAsset, onSelectAsset }: WatchlistPanelProps) {
  const results = useQueries({
    queries: assets.map((asset) => ({
      queryKey: ["watchlist", asset],
      queryFn: () => fetchWatchlistEntry(asset),
      staleTime: 60_000,
    })),
  });

  return (
    <div className="watchlist-panel">
      <span className="watchlist-panel__label">
        Mercado
        <InfoTooltip text={WATCHLIST_HELP} placement="bottom" />
      </span>
      <div className="watchlist-panel__list">
        {assets.map((asset, index) => {
          const result = results[index];
          const entry = result.data;
          const isActive = asset === activeAsset;

          return (
            <button
              key={asset}
              type="button"
              className={`watchlist-item${isActive ? " watchlist-item--active" : ""}`}
              onClick={() => onSelectAsset(asset)}
              aria-pressed={isActive}
            >
              <span className="watchlist-item__top-row">
                <span className="watchlist-item__asset">{asset}</span>
                {entry?.desactualizado && (
                  <span className="watchlist-item__stale-dot" title="Datos desactualizados" />
                )}
              </span>
              {result.isLoading && <span className="watchlist-item__loading">Cargando…</span>}
              {result.isError && <span className="watchlist-item__error">Sin datos</span>}
              {entry && entry.price !== null && (
                <>
                  <span className="watchlist-item__price">{formatPrice(entry.price)}</span>
                  <span
                    className={`watchlist-item__change${
                      entry.changePct !== null && entry.changePct < 0
                        ? " watchlist-item__change--down"
                        : " watchlist-item__change--up"
                    }`}
                  >
                    {entry.changePct !== null ? formatChangePct(entry.changePct) : "—"}
                  </span>
                </>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
