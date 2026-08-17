# crypto-quant-desk — frontend

Frontend web de crypto-quant-desk (Fase 8b): React + Vite + TypeScript,
gráfico de velas interactivo con [lightweight-charts](https://tradingview.github.io/lightweight-charts/)
(la librería gratuita de TradingView). Consume la API REST de
[`../api/`](../api/) (Fase 8a) — no calcula nada por su cuenta, ni
duplica lógica del backend Python: cada dato que se ve acá viene de un
endpoint (`/api/assets`, `/api/ohlcv`, `/api/studies`).

Corre en paralelo a la terminal de escritorio (`../app/`, PySide6) — son
dos frontends independientes sobre el mismo backend, ninguno reemplaza al
otro (todavía).

## Requisitos

- Node.js 20+ (probado con Node 22).
- La API corriendo (ver [`../api/main.py`](../api/main.py)):

  ```bash
  # desde la raíz del repo, con el entorno virtual de Python activado
  uvicorn api.main:app --reload
  ```

  Por defecto queda en `http://127.0.0.1:8000` (Swagger en `/docs`).

**Atajo (Fase 8d):** `../scripts/start.bat` (Windows) o `../scripts/start.sh`
(Mac/Linux) levantan la API y este frontend juntos, con un solo comando —
ver la sección "Arranque rápido" del [README de la raíz](../README.md).
Lo que sigue en esta sección es para correrlo a mano, paso a paso.

## Cómo correrlo

```bash
npm install
npm run dev
```

Abre `http://localhost:5173` (o el puerto que indique Vite). Necesita la
API corriendo para traer datos — si no la encuentra, la UI lo avisa con un
mensaje de error claro (no se queda cargando en silencio).

## Apuntar a otra URL de API

La URL base de la API es configurable vía la variable de entorno
`VITE_API_BASE_URL` (default: `http://127.0.0.1:8000`, ver
[`src/api.ts`](src/api.ts)). Para cambiarla:

```bash
cp .env.example .env.local
# editar .env.local con la URL que corresponda
```

`.env.local` queda fuera de git (`.gitignore`), así cada quien apunta a su
propia instancia sin pisar la de otro.

## Build de producción

```bash
npm run build    # tsc -b && vite build -> carpeta dist/
npm run preview  # sirve el build de dist/ para probarlo localmente
```

## Estructura

```
src/
  api.ts                        # cliente HTTP de la API (capa fina, sin cálculos)
  types.ts                      # tipos TS que reflejan los esquemas Pydantic de la API
  theme.ts                      # paleta de colores del gráfico (misma que app/theme.py)
  alerts.ts                     # tipos + evaluación pura de reglas de alerta (Fase 8d)
  drawings.ts                   # tipos de dibujos sobre el gráfico (Fase 8d)
  App.tsx                       # estado compartido (activo/timeframe/vista activa)
  hooks/
    useLocalStorageState.ts     # persistencia genérica en localStorage (Fase 8d)
  components/
    AssetSelector.tsx           # selector de activo + timeframe
    StudyToggles.tsx            # checkboxes de overlays sobre el precio (SMA/EMA/Bollinger/Fibonacci/S-R/Pivotes)
    OscillatorPanel.tsx         # checkboxes de osciladores (RSI/MACD/Estocástico)
    Chart.tsx                   # gráfico de velas (lightweight-charts) + overlays + panes de osciladores
    SuggesterPanel.tsx          # panel del sugeridor de consenso (Fase 8c)
    AlertsPanel.tsx             # reglas de alerta client-side + toasts (Fase 8d)
    DrawingTools.tsx            # líneas horizontales/de tendencia sobre el gráfico (Fase 8d)
  views/
    TechnicalAnalysisView.tsx   # velas + overlays + osciladores + sugeridor + alertas + dibujo
    RiskView.tsx
    BacktestView.tsx
    ResearchView.tsx
```

## Alcance actual (hasta Fase 8d)

Las 4 vistas (Análisis Técnico, Riesgo, Backtest, Research) y todo lo
descrito arriba. La app de escritorio (PySide6, `../app/`) sigue
existiendo en paralelo — ninguna reemplaza a la otra (todavía).

Por diseño, al entrar a Análisis Técnico solo se muestran SMA 20 y SMA 50
sobre las velas y ningún oscilador — el resto de los overlays/osciladores
se agregan desde los checkboxes, nunca todo encimado de entrada.

**Alertas (Fase 8d) — limitación honesta:** las reglas se evalúan
client-side, en el navegador, contra los datos que ya trae `/api/studies`.
Solo disparan mientras la pestaña está abierta y la vista de Análisis
Técnico cargada — no hay notificaciones push, email, ni nada que llegue
con la app cerrada. Las reglas sí persisten en `localStorage` entre
sesiones.

**Dibujo (Fase 8d) — enfoque técnico:** `lightweight-charts` (versión
gratuita, la que usa este proyecto) no trae herramientas de dibujo
interactivas — eso es parte de un plugin comercial aparte. Líneas
horizontales usan la primitiva nativa `series.createPriceLine()` (la
misma que ya se usaba para Fibonacci/soporte-resistencia/pivotes). Líneas
de tendencia se arman a mano con una `LineSeries` de 2 puntos, tomando el
click del usuario sobre el gráfico vía `chart.subscribeClick()` +
`series.coordinateToPrice()` — no hay dependencia nueva, todo con la API
pública gratuita de la librería. Los dibujos persisten en `localStorage`,
por activo y timeframe.
