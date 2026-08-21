/**
 * Botón "Descargar informe PDF" (Fase 16b) — dispara `/api/report` (MUY
 * LENTO: ajusta un GARCH y corre cointegración rolling sobre todos los
 * pares, del orden de 30-90 segundos) SOLO ante un click explícito
 * (`enabled: false` + `refetch()`, mismo patrón que `ResearchView` para
 * `/api/prediction` — nunca se dispara solo al montar). Al terminar,
 * dispara la descarga del archivo en el navegador vía un link temporal
 * con `URL.createObjectURL` (la respuesta es un `Blob`, no JSON).
 */

import { useQuery } from "@tanstack/react-query";
import { ApiError, getPdfReport } from "../api";

interface ReportDownloadButtonProps {
  asset: string;
  interval: string;
}

function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function ReportDownloadButton({ asset, interval }: ReportDownloadButtonProps) {
  const reportQuery = useQuery({
    queryKey: ["report", asset, interval],
    queryFn: () => getPdfReport(asset, interval),
    enabled: false,
    retry: 0,
  });

  const error = reportQuery.error;
  const errorMessage = error instanceof ApiError ? error.message : error ? String(error) : null;

  const handleClick = () => {
    void reportQuery.refetch().then((result) => {
      if (result.data) {
        triggerBlobDownload(result.data, `informe_${asset}_${interval}.pdf`);
      }
    });
  };

  return (
    <div className="data-status-bar">
      <div className="data-status-bar__info">
        <span className="view-note">
          Informe PDF con gráficas y explicaciones de riesgo, técnico, ciclos, correlación y arbitraje para {asset}.
        </span>
      </div>

      <button type="button" className="secondary-button" onClick={handleClick} disabled={reportQuery.isFetching}>
        {reportQuery.isFetching && <span className="spinner spinner--inline" />}
        {reportQuery.isFetching ? "Generando informe…" : "Descargar informe PDF"}
      </button>

      {reportQuery.isFetching && (
        <span className="data-status-bar__fresh">Puede tardar 30-90 segundos — no cierres esta página.</span>
      )}
      {errorMessage && <span className="data-status-bar__error">{errorMessage}</span>}
    </div>
  );
}
