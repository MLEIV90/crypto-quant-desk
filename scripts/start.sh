#!/usr/bin/env bash
#
# crypto-quant-desk - arranque de un click (Mac/Linux)
#
# A diferencia de start.bat (que abre 2 ventanas de consola en Windows),
# acá backend y frontend corren en SEGUNDO PLANO del mismo terminal: no
# hay forma portable de "abrir una ventana nueva" que funcione igual en
# Terminal.app, iTerm2, gnome-terminal, xterm, etc. sin asumir cuál tiene
# el usuario instalado. Ctrl+C en este terminal mata a los dos procesos
# (ver trap más abajo).
#
# Requiere: entorno virtual de Python en .venv/ (ver README) y Node.js/npm
# instalados y en el PATH.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== crypto-quant-desk: arranque ==="
echo

PYTHON_BIN="$ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "[ERROR] No se encontró el entorno virtual en .venv/bin/python"
  echo "        Crealo con:"
  echo "            python3 -m venv .venv"
  echo "            .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[ERROR] No se encontró Node.js en el PATH."
  echo "        Instalalo desde https://nodejs.org/ y volvé a intentar."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[ERROR] No se encontró npm en el PATH (debería venir con Node.js)."
  echo "        Reinstalá Node.js desde https://nodejs.org/"
  exit 1
fi

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "Primera vez: instalando dependencias del frontend con 'npm install'..."
  echo "(esto puede tardar un par de minutos)"
  echo
  if ! (cd "$ROOT/frontend" && npm install); then
    echo
    echo "[ERROR] 'npm install' falló. Revisá el mensaje de arriba."
    exit 1
  fi
  echo
fi

API_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "Cerrando backend y frontend..."
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Levantando backend (API) en http://127.0.0.1:8000 ..."
"$PYTHON_BIN" -m uvicorn api.main:app --reload &
API_PID=$!

echo "Levantando frontend (Vite) en http://localhost:5173 ..."
(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

echo "Esperando unos segundos a que el frontend termine de levantar..."
sleep 6

URL="http://localhost:5173"
if command -v open >/dev/null 2>&1; then
  open "$URL" 2>/dev/null            # macOS
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" 2>/dev/null        # Linux
else
  echo "No se pudo abrir el navegador automáticamente — abrí manualmente: $URL"
fi

echo
echo "Listo. Backend y frontend corriendo en este terminal."
echo "Si el navegador no muestra nada todavía, esperá unos segundos más y"
echo "recargá — o revisá el log de arriba por si Vite eligió otro puerto"
echo "(ej. 5174) porque el 5173 estaba ocupado."
echo
echo "Presioná Ctrl+C para detener backend y frontend."
echo

wait
