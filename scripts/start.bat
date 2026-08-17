@echo off
setlocal

:: crypto-quant-desk - arranque de un click (Windows)
:: Levanta el backend (uvicorn) y el frontend (Vite) cada uno en su propia
:: ventana de consola, espera a que el frontend levante y abre el navegador.
:: Requiere: entorno virtual de Python en .venv\ (ver README) y Node.js/npm
:: instalados y en el PATH.

set "ROOT=%~dp0.."
cd /d "%ROOT%"

echo === crypto-quant-desk: arranque ===
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en .venv\Scripts\python.exe
    echo         Crealo con:
    echo             python -m venv .venv
    echo             .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro Node.js en el PATH.
    echo         Instalalo desde https://nodejs.org/ y volve a intentar.
    echo.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No se encontro npm en el PATH ^(deberia venir con Node.js^).
    echo         Reinstala Node.js desde https://nodejs.org/
    echo.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo Primera vez: instalando dependencias del frontend con "npm install"...
    echo ^(esto puede tardar un par de minutos^)
    echo.
    pushd frontend
    call npm install
    set "NPM_INSTALL_RESULT=%errorlevel%"
    popd
    if not "%NPM_INSTALL_RESULT%"=="0" (
        echo.
        echo [ERROR] "npm install" fallo. Revisa el mensaje de arriba.
        echo.
        pause
        exit /b 1
    )
    echo.
)

echo Levantando backend ^(API^) en http://127.0.0.1:8000 ...
start "crypto-quant-desk - API" cmd /k "cd /d "%ROOT%" && .venv\Scripts\python.exe -m uvicorn api.main:app --reload"

echo Levantando frontend ^(Vite^) en http://localhost:5173 ...
start "crypto-quant-desk - frontend" cmd /k "cd /d "%ROOT%\frontend" && npm run dev"

echo Esperando unos segundos a que el frontend termine de levantar...
timeout /t 6 /nobreak >nul

start "" "http://localhost:5173"

echo.
echo Listo. Se abrieron 2 ventanas ^(API y frontend^) y el navegador en
echo http://localhost:5173
echo.
echo Si el navegador no muestra nada todavia, esperá unos segundos mas y
echo recargá — o revisa la ventana "crypto-quant-desk - frontend" por si
echo Vite eligio otro puerto ^(ej. 5174^) porque el 5173 estaba ocupado.
echo.
echo Para cerrar todo: cerrá las 2 ventanas de consola que se abrieron.
echo.
pause
