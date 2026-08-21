/**
 * Dispara la descarga de un `Blob` en el navegador vía un link temporal
 * (Fase 16b, factorizado en Fase 17a para compartirlo entre el informe PDF
 * y las exportaciones CSV) — `URL.createObjectURL` + click programático,
 * sin depender de que el backend setee `Content-Disposition` para que el
 * navegador dispare el diálogo de "Guardar como" (ya lo hace, pero el link
 * con `download=filename` es lo que dispara la descarga real en vez de
 * abrir el archivo en una pestaña nueva).
 */
export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
