/**
 * Beep corto para alertas disparadas (Fase 13c) — generado con Web Audio
 * API (oscilador simple), sin archivo de audio nuevo ni dependencia extra.
 *
 * Los navegadores suspenden el audio hasta la primera interacción del
 * usuario con la página (política de autoplay) — si el beep no suena la
 * primera vez que se dispara una alerta, es por eso, no un bug; alguna
 * interacción previa con la página (un click en cualquier lado) lo
 * desbloquea para el resto de la sesión.
 */

let sharedContext: AudioContext | null = null;

function getContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const AudioCtx =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioCtx) return null;
  if (!sharedContext) sharedContext = new AudioCtx();
  return sharedContext;
}

export function playAlertBeep(): void {
  const ctx = getContext();
  if (!ctx) return;
  try {
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
    oscillator.connect(gain);
    gain.connect(ctx.destination);
    oscillator.start();
    oscillator.stop(ctx.currentTime + 0.25);
  } catch {
    // El audio puede fallar en contextos restringidos (ej. sin gesto de
    // usuario todavía) — no es crítico, la alerta ya se mostró igual
    // (toast + historial + resaltado en el watchlist).
  }
}
