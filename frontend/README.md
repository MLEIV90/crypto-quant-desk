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
  App.tsx                       # estado (activo/timeframe/toggles) + fetch vía React Query
  components/
    AssetSelector.tsx           # selector de activo + timeframe
    StudyToggles.tsx            # checkboxes de overlays sobre el precio (SMA/EMA/Bollinger/Fibonacci/S-R/Pivotes)
    OscillatorPanel.tsx         # checkboxes de osciladores (RSI/MACD/Estocástico)
    Chart.tsx                   # gráfico de velas (lightweight-charts) + overlays + panes de osciladores
```

## Alcance de esta fase (8b)

Solo el gráfico interactivo: velas + overlays toggleables + osciladores en
panes sincronizados. **Todavía no incluye** las pestañas de riesgo/backtest
ni el sugeridor de consenso (Fase 8c), ni alertas/dibujo manual sobre el
gráfico (Fase 8d) — la app de escritorio (PySide6) sigue siendo, por ahora,
la única forma de ver esas partes del proyecto.

Por diseño, al entrar solo se muestran SMA 20 y SMA 50 sobre las velas y
ningún oscilador — el resto de los overlays/osciladores se agregan desde
los checkboxes, nunca todo encimado de entrada.
