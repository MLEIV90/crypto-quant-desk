/**
 * Botón "Descargar CSV" genérico (Fase 17a) — mismo patrón que
 * `ReportDownloadButton` (Fase 16b) para el informe PDF: dispara el fetch
 * SOLO ante un click explícito (`enabled: false` + `refetch()`, nunca al
 * montar), y al terminar dispara la descarga real del archivo vía
 * `triggerBlobDownload`. A diferencia del informe PDF, la exportación a
 * CSV es rápida (mismos cálculos ya livianos que sus vistas JSON
 * hermanas) — no hace falta un aviso de demora, alcanza con el spinner.
 *
 * Genérico en `fetchCsv`/`queryKey` para no repetir esta lógica en cada
 * vista (Análisis Técnico, Riesgo, Ciclos y Estadística, Correlación) —
 * cada vista solo pasa QUÉ pedir y cómo llamar al archivo resultante.
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError } from "../api";
import { triggerBlobDownload } from "../downloadBlob";

interface CsvDownloadButtonProps {
  label: string;
  filename: string;
  fetchCsv: () => Promise<Blob>;
  queryKey: unknown[];
}

export function CsvDownloadButton({ label, filename, fetchCsv, queryKey }: CsvDownloadButtonProps) {
  const query = useQuery({
    queryKey,
    queryFn: fetchCsv,
    enabled: false,
    retry: 0,
  });

  const error = query.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  const handleClick = () => {
    void query.refetch().then((result) => {
      if (result.data) {
        triggerBlobDownload(result.data, filename);
      }
    });
  };

  return (
    <span className="csv-download">
      <button type="button" className="secondary-button" onClick={handleClick} disabled={query.isFetching}>
        {query.isFetching && <span className="spinner spinner--inline" />}
        {query.isFetching ? "Generando CSV…" : label}
      </button>
      {errorMessage && <span className="data-status-bar__error">{errorMessage}</span>}
    </span>
  );
}
