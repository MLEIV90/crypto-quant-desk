/**
 * Textos de ayuda centralizados (Fase 9b) — un solo lugar para mantener y
 * eventualmente traducir todas las explicaciones que muestra `InfoTooltip`
 * en la app. Ningún componente hardcodea su propio texto de ayuda.
 *
 * CRITERIO DE REDACCIÓN, el mismo que ya rige el resto del proyecto (ver
 * `signals/suggester.py`, `ml/models.py`): para los estudios SIN edge
 * demostrado (que son la mayoría — Fibonacci, soporte/resistencia,
 * pivotes, y en general cualquier indicador técnico clásico), el texto
 * dice explícitamente que no predice y que no tiene respaldo estadístico
 * probado en este proyecto. Nada de lenguaje que sugiera que un estudio
 * "anticipa" o "confirma" un movimiento futuro.
 */

import type { OverlayKey } from "./components/StudyToggles";
import type { OscillatorKey } from "./components/OscillatorPanel";

export const OVERLAY_HELP: Record<OverlayKey, string> = {
  sma20: "Promedio simple del precio de cierre de las últimas 20 velas. Suaviza el ruido de corto plazo para ver hacia dónde viene la tendencia — no anticipa hacia dónde va.",
  sma50: "Igual que la SMA 20 pero con 50 velas: más lenta para reaccionar, menos ruido. Se usa como referencia de tendencia de más largo plazo.",
  ema12: "Promedio exponencial de 12 velas: a diferencia de la SMA, pondera más los precios recientes. Es una de las dos medias que arman el MACD.",
  ema26: "Promedio exponencial de 26 velas. Junto con la EMA 12 forma la línea del MACD (ver el oscilador MACD).",
  bollinger: "Banda de ±2 desvíos estándar alrededor de una media móvil. Cuando se abre hubo más volatilidad reciente; cuando se cierra, menos. No indica en qué dirección va a romper el precio.",
  fibonacci: "Niveles de retroceso/extensión populares entre traders técnicos, calculados sobre un máximo y un mínimo reciente. SIN respaldo estadístico probado en este proyecto — es una referencia visual, no una regla.",
  supportResistance: "Zonas de precio donde el activo rebotó o se frenó recientemente. Que haya pasado antes no garantiza que vuelva a pasar — es una referencia visual, no una predicción.",
  pivots: "Niveles calculados a partir del rango (alto/bajo/cierre) de la vela anterior. Muy usados en trading de corto plazo; el precio puede perforarlos sin aviso.",
  vwap: "Precio promedio ponderado por volumen (ventana móvil de 20 velas). Referencia de 'valor justo' reciente — no predice hacia dónde va el precio.",
  ichimoku: "Sistema de líneas y una 'nube' que algunos traders usan para leer tendencia y soporte/resistencia. Popular, pero sin respaldo estadístico de edge probado en este proyecto.",
  volumeProfile:
    "Muestra CUÁNTO volumen se operó en cada NIVEL DE PRECIO del período (a diferencia del volumen normal, que es por vela/tiempo) — el histograma horizontal a la derecha del gráfico. El POC (Point of Control, resaltado) es el nivel con más volumen; el Value Area (sombreado) es el rango que concentra el 70% del volumen. Los niveles de alto volumen SUELEN actuar como soporte/resistencia porque mucha gente operó ahí, pero no es una regla garantizada: el precio puede atravesarlos sin reaccionar. Se recalcula sobre el período elegido arriba (1M/3M/1A/etc.), no sobre todo el histórico.",
};

export const OSCILLATOR_HELP: Record<OscillatorKey, string> = {
  rsi: "Mide si el activo está sobrecomprado (>70) o sobrevendido (<30) según la velocidad y magnitud de los cambios de precio recientes. No predice: muestra momentum reciente, nada más.",
  macd: "Diferencia entre la EMA 12 y la EMA 26 del precio, con su propia media de señal. Un cruce entre la línea MACD y su señal se lee como cambio de momentum — no es una señal de compra/venta garantizada.",
  stochastic: "Compara el cierre actual contra el rango de precios reciente, en una escala de 0 a 100. Por encima de 80 sugiere sobrecompra, por debajo de 20 sobreventa — mide momentum, igual que el RSI, no anticipa el precio.",
  obv: "Acumula el volumen sumándolo o restándolo según el precio suba o baje. Se usa para ver si el volumen acompaña al movimiento del precio — no anticipa el precio en sí.",
};

export const CHART_TYPE_HELP =
  "Heikin-Ashi suaviza las velas promediando cada una con la anterior, para que las tendencias se vean más " +
  "limpias. El open/high/low/close de cada vela deja de ser el precio real negociado — para ver el precio " +
  "exacto, volvé a 'Velas'.";

export const RISK_INTRO_HELP =
  "Estas son medidas de RIESGO calculadas sobre la historia del activo — ninguna predice el precio de mañana. " +
  "Sirven para decidir CUÁNTO exponerte (tamaño de posición), no CUÁNDO vas a acertar.";

export const RISK_METRIC_HELP = {
  volRealizada:
    "Qué tan movido estuvo el precio en el pasado reciente (volatilidad histórica, anualizada). Más alta = " +
    "movimientos diarios más grandes, en ambas direcciones.",
  modeloGarch:
    "El modelo estadístico (familia GARCH) que mejor ajustó la volatilidad histórica de este activo, elegido " +
    "automáticamente por criterio AIC — no lo elige un humano a mano.",
  volGarch:
    "Estimación de volatilidad de ese modelo GARCH: le da más peso a los movimientos recientes que la volatilidad " +
    "realizada simple. Es una proyección de corto plazo sobre datos pasados, no una certeza sobre el futuro.",
  regimen:
    "CALMA = volatilidad baja respecto a la propia historia del activo. NORMAL = dentro de lo típico. TENSIÓN = " +
    "volatilidad alta. Útil para ajustar el tamaño de posición, no para adivinar el momento de entrar o salir.",
  var95:
    "En un día malo típico (el peor 5% de los casos históricos), podrías perder alrededor de esto. Es una medida " +
    "de riesgo estadística sobre el pasado, no una predicción de lo que va a pasar mañana.",
  es95:
    "Si ese día malo (el peor 5%) efectivamente ocurre, esto es la pérdida PROMEDIO esperada en ese escenario — " +
    "siempre es igual o mayor al VaR, porque mira más adentro de la cola mala de la distribución.",
  senal:
    "Dirección sugerida por el motor de señales (tendencia + momentum + reversión a la media combinados en un " +
    "score). Es un dato de apoyo, no una recomendación de inversión.",
  sizing:
    "Tamaño de posición sugerido por 'vol targeting': ajusta la exposición para apuntar a una volatilidad objetivo " +
    "del book, reduciendo el tamaño cuando el activo está más volátil de lo normal.",
};

// Fase 20a: percentil histórico, histograma de retornos y franja de régimen.

export const RISK_PERCENTILE_HELP =
  "Percentil histórico de esta métrica: qué fracción de TODA la historia del activo tuvo un valor menor o " +
  "igual al de hoy. Un percentil de 90 significa 'hoy está más alto que el 90% de los días históricos' — es " +
  "una foto descriptiva de qué tan inusual es el valor de hoy respecto de su propio pasado, no una " +
  "probabilidad ni una predicción de lo que va a pasar mañana. VaR/ES comparan contra una ventana móvil de " +
  "1 año (no contra el número único de toda la historia que se muestra arriba), porque necesitan su propia " +
  "serie para poder ubicar el valor de hoy en algún lado.";

export const RISK_HISTOGRAM_HELP =
  "Distribución de los retornos diarios del activo. El VaR marca la pérdida del peor 5% de los días; el ES, " +
  "el promedio de pérdida cuando ese 5% peor efectivamente ocurre (por eso el ES siempre cae más a la " +
  "izquierda, más adentro de la cola mala). Las barras de esa cola se resaltan para que se vea, literalmente, " +
  "dónde vive el riesgo — no es una curva teórica (normal/gaussiana), es la distribución REAL observada.";

export const RISK_REGIME_STRIP_HELP =
  "Régimen de volatilidad (calma/normal/tensión, ver más arriba) de CADA fecha, no solo la de hoy — así se ve " +
  "cómo alternó en la historia del activo y si el momento actual es parte de un tramo largo o un cambio " +
  "reciente. Es una clasificación retrospectiva sobre datos ya conocidos, no un aviso de que el régimen esté " +
  "por cambiar.";

// Fase 20b: vol targeting comparado contra buy & hold, y tabla de riesgo de las 5 monedas.

export const RISK_VOL_TARGETING_HELP =
  "Vol targeting, en criollo: en vez de invertir siempre el mismo tamaño, reduce la exposición cuando el " +
  "activo está más volátil de lo normal y la aumenta cuando está más tranquilo — apunta a que el RIESGO de " +
  "la cartera sea parejo en el tiempo, no el capital invertido. SIEMPRE está comprado, nunca adivina la " +
  "dirección: por eso, solo, reduce el drawdown de forma modesta (sigue expuesto en las caídas, nada más " +
  "que con menos tamaño). La columna 'Engine + vol targeting' agrega la señal direccional del motor de " +
  "señales encima de ese mismo sizing — esa combinación es la que puede salirse del mercado en una caída, " +
  "y por eso el drawdown baja mucho más ahí. Mostrar las tres por separado (Fase 22) es a propósito: antes " +
  "esta sección le atribuía el resultado de la combinación al vol targeting solo, algo que no era cierto.";

export const RISK_SUMMARY_HELP =
  "Comparación de riesgo actual entre las 5 monedas — cuál está más volátil o en tensión hoy. El VaR de esta " +
  "tabla (Fase 25) es EMPÍRICO en ventana móvil de 1 año, el MISMO método y el MISMO número que el 'VaR 95% " +
  "actual' del panel de abajo para el activo seleccionado — antes eran métodos distintos y podían mostrar " +
  "números distintos para la misma moneda, ya no. El RÉGIMEN sí sigue siendo dos métodos a propósito: acá usa " +
  "volatilidad REALIZADA (rápida, sin ajustar un GARCH por moneda) mientras el panel de abajo usa GARCH — " +
  "ajustar 5 modelos GARCH a la vez sería demasiado lento para una tabla que se lee de un vistazo, así que " +
  "puede diferir del régimen del panel de abajo sin que eso sea una contradicción: el rótulo de la columna " +
  "aclara la base exacta. Hacé clic en una fila para pasar a ver el detalle completo de esa moneda.";

// Fase 20c: coherencia del panel de riesgo — VaR/ES "actual" (GARCH, hoy)
// vs. "histórico" (toda la serie), y bases de comparación explícitas.

export const RISK_ACTUAL_VS_HISTORICO_HELP =
  "Antes, el VaR y el ES que se mostraban acá eran un único número calculado sobre TODA la historia del " +
  "activo — el mismo valor tanto en un día de calma como en uno de tensión, así que no servían para " +
  "responder '¿cuánto riesgo hay HOY?'. El VaR/ES ACTUAL resuelve eso: se calcula de forma EMPÍRICA sobre " +
  "una ventana móvil del último año (el mismo método, sin GARCH, que ya usa la tabla de las 5 monedas de " +
  "arriba — Fase 25, antes este panel usaba en cambio un método paramétrico vía GARCH que daba un número " +
  "distinto al de esa tabla para la misma moneda), así que sube y baja junto con el régimen reciente. El " +
  "histórico se conserva como referencia de largo plazo (útil para comparar activos entre sí en un período " +
  "largo), pero rotulado aparte para no confundirlo con 'el riesgo de hoy'. La volatilidad condicional GARCH " +
  "no se pierde: sigue mostrándose como métrica de volatilidad y régimen, más arriba.";

export const SUGGESTER_HELP =
  "Es un VOTO de varios estudios técnicos (RSI, medias, MACD, estocástico, Bollinger, pivote) combinados por " +
  "mayoría — no un modelo entrenado ni una IA que aprendió patrones. Antes de hacerle caso, mirá su desempeño " +
  "histórico (Sharpe/CAGR/drawdown) de acá abajo: si no le gana con claridad al buy & hold, tratalo como un dato " +
  "más entre varios, no como una orden de operar.";

export const BACKTEST_EQUITY_HELP =
  "Evolución de $1 invertido desde el inicio del período, para la estrategia elegida y para comprar-y-mantener " +
  "en paralelo. Muestra CÓMO se llegó al resultado final (con qué subidas y caídas en el camino), no solo el " +
  "número final. Usá la escala logarítmica si una de las dos curvas aplasta visualmente a la otra.";

export const BACKTEST_LOG_SCALE_HELP =
  "En escala LINEAL, una curva que llegó mucho más alto (p. ej. buy & hold en un mercado alcista largo) aplasta " +
  "visualmente a la otra, aunque ambas hayan tenido tramos interesantes. En escala LOGARÍTMICA, la misma variación " +
  "porcentual ocupa el mismo espacio vertical en cualquier nivel — así se puede comparar el COMPORTAMIENTO de las " +
  "dos curvas (qué tan volátil fue cada una, dónde cayó cada una) y no solo cuál terminó más arriba.";

export const BACKTEST_DRAWDOWN_HELP =
  "El gráfico de arriba (equity) muestra hacia dónde fue el capital; este muestra cuánto se estaba perdiendo en " +
  "CADA momento respecto del pico anterior — 0% en un máximo nuevo, más negativo cuanto más profunda la caída en " +
  "curso. Es la vista 'bajo el agua' (underwater) del drawdown: donde la curva de una estrategia se mantiene cerca " +
  "de 0% casi todo el tiempo y la de la otra pasa largos tramos muy abajo, ESO es la diferencia real entre sufrir " +
  "una caída del 15% y sufrir una del 80%, aunque el retorno final de ambas fuera parecido.";

export const BACKTEST_STRATEGY_SELECTOR_HELP =
  "Las tres estrategias reutilizan las mismas piezas del motor de señales (`signals.engine`) combinadas de forma " +
  "distinta — ninguna es un modelo nuevo entrenado para esta vista. Comparalas contra buy & hold (el control: " +
  "siempre 100% invertido, sin señal de ningún tipo) para ver si de verdad aportan algo, y entre sí para ver qué " +
  "problema resuelve cada una.";

export const BACKTEST_EXPOSURE_HELP =
  "Fase 23: los gráficos de arriba muestran el RESULTADO (equity, drawdown); este muestra el COMPORTAMIENTO — " +
  "cuánto invertida estuvo la estrategia en cada fecha, con el mismo desfase de un día que ya aplica el " +
  "backtest (la decisión de HOY recién es efectiva mañana). Vol targeting se mueve entre 0 (afuera, típicamente " +
  "solo durante el arranque) y 1 (100% largo, nunca corto); las estrategias con señal direccional (engine, " +
  "combinado) pueden bajar hasta -1 (100% corto) o pasar por 0 (afuera del mercado) — ver EN QUÉ MOMENTOS pasó " +
  "cada cosa es lo que este gráfico agrega sobre las métricas resumidas de la tabla.";

export const BACKTEST_PARAMS_HELP =
  "El costo de transacción (en basis points, 1 bp = 0.01%) se cobra sobre cada cambio de tamaño de posición — " +
  "estrategias que operan más seguido son más sensibles a subirlo. El rango de fechas recorta la VENTANA del " +
  "backtest después de calcular la señal sobre todo el histórico disponible, así que no genera un 'arranque en " +
  "frío' artificial al principio del rango elegido.";

export const WATCHLIST_HELP =
  "Precio de cierre más reciente y variación % contra la vela diaria anterior de cada activo. Es el cambio " +
  "entre las últimas DOS velas guardadas, no necesariamente las últimas 24hs de reloj si los datos no están " +
  "recién actualizados — mirá el punto ámbar y 'Actualizar datos' arriba.";

export const BACKTEST_METRIC_HELP: Record<string, string> = {
  cagr: "Crecimiento anual compuesto: a qué tasa anual equivalente creció el capital en promedio durante el período. Más alto es mejor.",
  sharpe: "Retorno ajustado por riesgo: cuánto retorno se obtuvo por unidad de volatilidad asumida. Más alto es mejor; no distingue una caída brusca de una subida brusca (ver Sortino).",
  sortino: "Como el Sharpe, pero solo penaliza la volatilidad A LA BAJA — las subidas bruscas no lo perjudican. Más alto es mejor.",
  max_drawdown: "La peor caída, de punta a punta, desde un máximo hasta el mínimo posterior en todo el período — cuánto tuviste que aguantar en el peor momento. Se expresa como número negativo: más cerca de 0% es mejor (menos dolor).",
  calmar: "Retorno anualizado dividido por el máximo drawdown: cuánto retorno obtuviste por cada unidad de 'dolor' de la peor caída. Más alto es mejor.",
  exposicion_media: "Fase 23: el tamaño de posición PROMEDIO en valor absoluto a lo largo de todo el período (100% = siempre a full, 0% = siempre afuera). Complementa a 'cambios de dirección': vol targeting puede mostrar 1 cambio de dirección Y una exposición promedio bien por debajo del 100% si reduce tamaño seguido. Ni mejor ni peor por sí sola.",
  pct_tiempo_fuera: "Fase 23: qué fracción de los días la posición efectiva fue EXACTAMENTE cero (afuera del mercado del todo) — incluye el primer día (sin decisión previa) y el arranque de estrategias con ventanas de cálculo. Una estrategia siempre invertida (buy & hold, vol targeting) muestra un número casi nulo acá; una con señal direccional que sale del mercado seguido (engine, combinado) muestra bastante más.",
  n_trades: "Fase 22: cuántas veces la estrategia cambió de SIGNO (pasó de estar afuera/corta a larga, o viceversa) — NO cuenta los ajustes de TAMAÑO dentro de la misma dirección, eso lo refleja el turnover de la fila de abajo. Una estrategia que siempre está comprada (p. ej. vol targeting puro) puede mostrar 1 acá y aun así rebalancear todos los días: nunca cambia de signo, pero sí cambia de tamaño constantemente. Ni mejor ni peor por sí solo.",
  turnover_total: "Cuánto 'movimiento' de posición hubo en total — más turnover implica más costos de transacción reales. Más bajo es mejor en igualdad de retorno, no es un fin en sí mismo.",
};

export const PERIOD_SELECTOR_HELP =
  "Elegí cuánta historia mostrar. 'Todo' trae el histórico completo — en velas horarias eso son unas 58.000 " +
  "velas y puede tardar 1-2 segundos en cargar. Es esperado, no un error.";

// --------------------------------------------------------------------------
// Vista "Ciclos y Estadística" (Fase 11, rehecha en Fase 15a) — /api/stats.
// Mismo criterio de honestidad que el resto: nada de esto predice el
// precio, se explicita en cada bloque.
// --------------------------------------------------------------------------

export const STATS_INTRO_HELP =
  "Estos análisis describen patrones HISTÓRICOS del activo (estacionalidad, autocorrelación, drawdowns, " +
  "fases de mercado) — ninguno predice el precio de mañana. En cripto, con pocos años de historia " +
  "disponible, tratalos como observaciones curiosas para tu propio criterio, no como reglas para operar.";

export const SEASONALITY_HELP = {
  weekday:
    "Retorno promedio histórico agrupando por día de la semana (y por hora UTC, si el timeframe es horario). " +
    "A diferencia de las acciones, cripto opera los 7 días — no hay 'efecto fin de semana' por falta de " +
    "mercado abierto. Patrón débil e inestable, no una regla operable. La estacionalidad MENSUAL está más " +
    "abajo, en el mapa de calor mes x año — ahí se ve año por año, no como un promedio único.",
  noiseCaveatIntro:
    "Fase 26: que un día promedie más que otro NO significa que ese día 'convenga' — compará esa diferencia " +
    "contra cuánto varía el retorno de un día cualquiera (el desvío estándar) para ver si hay una señal real " +
    "o solo ruido.",
};

export const STATIONARITY_HELP =
  "El test ADF (Augmented Dickey-Fuller) pregunta si una serie 'vuelve' hacia un nivel estable con el tiempo " +
  "(estacionaria) o si puede alejarse sin límite (no estacionaria, como un 'paseo aleatorio'). El PRECIO de " +
  "un activo cripto típicamente NO es estacionario (tiene tendencia de largo plazo) — por eso este proyecto " +
  "modela sobre RETORNOS, que sí suelen ser estacionarios (oscilan alrededor de una media estable). p-valor < " +
  "0.05 se lee como 'sí es estacionaria'.";

export const STATIONARITY_PLAIN_HELP =
  "En criollo: que el PRECIO no sea estacionario significa que deambula sin un nivel al que 'vuelva' — hoy " +
  "puede estar en cualquier lado según toda la tendencia acumulada, así que su nivel actual no sirve como " +
  "referencia para predecir nada. Que los RETORNOS sí lo sean significa que sus variaciones día a día tienen " +
  "una estructura ESTABLE en el tiempo (una media y una dispersión que no se van a cualquier lado) — por eso " +
  "todo el análisis serio de este proyecto (y en general) se hace sobre retornos, nunca sobre el precio crudo.";

export const AUTOCORRELATION_HELP = {
  returns:
    "Autocorrelación del NIVEL de retorno, rezago a rezago: ¿el retorno de hace N velas ayuda a predecir el " +
    "de hoy? En un mercado razonablemente eficiente, estas barras deberían estar cerca de 0 — eso es lo " +
    "ESPERADO, no una falla del análisis.",
  returnsPlain:
    "¿Para qué sirve esto? Si estas barras estuvieran lejos de 0 de forma consistente, se podría usar el " +
    "retorno de ayer (o de hace varios días) para tener una pista sobre el de hoy. Que estén pegadas a 0 en " +
    "todos los rezagos CONFIRMA cuantitativamente lo que encontró el resto de la investigación del proyecto " +
    "(ver la pestaña Research): la DIRECCIÓN del precio no se predice a partir de su propio pasado.",
  squared:
    "Autocorrelación de los retornos AL CUADRADO: mide si a un día volátil le sigue otro día volátil (aunque " +
    "el SIGNO del retorno no se pueda predecir). Positiva y persistente en los primeros rezagos es lo típico " +
    "en cripto — 'clustering de volatilidad', la motivación de modelar con GARCH (ver la vista Riesgo).",
  squaredPlain:
    "¿Para qué sirve esto? Que estas barras sean POSITIVAS y no caigan de golpe a 0 muestra que la " +
    "volatilidad se agrupa: días agitados tienden a seguir a otros días agitados, y días tranquilos a otros " +
    "tranquilos — aunque no se sepa hacia dónde va el precio. Esa es, en una frase, la base estadística de " +
    "por qué el RIESGO sí se puede gestionar (vol targeting, GARCH) aunque la DIRECCIÓN no se pueda predecir.",
};

// Fase 15a: reemplaza al periodograma (CYCLES_HELP) — sobre retornos
// diarios daba "ciclos" de 2-3 días, ruido de alta frecuencia sin ningún
// significado de mercado. Se reemplaza por ciclos que sí tienen un
// significado reconocible: drawdowns, fases bull/bear, y halvings.

export const DRAWDOWN_HELP =
  "Un drawdown es una caída desde el MÁXIMO HISTÓRICO de toda la serie hasta el mínimo posterior, antes de " +
  "volver a superar ese máximo. 'Profundidad' es cuánto cayó desde el pico; 'días de caída' cuánto tardó en " +
  "llegar al fondo; 'días de recuperación' cuánto tardó en volver a superar el pico anterior (vacío si " +
  "todavía no recuperó). Es historia, no una garantía de que el próximo drawdown vaya a tener una forma " +
  "parecida.";

export const MARKET_PHASES_HELP =
  "Acá una fase pasa a ser BEAR en cuanto el precio cae 20% o más desde un máximo LOCAL (el techo de esa " +
  "tendencia, no necesariamente el máximo histórico de toda la serie), y BULL en cuanto sube 20% o más desde " +
  "un mínimo local — una regla mecánica y arbitraria (el 20% es la convención que usan medios financieros, " +
  "no un número mágico). Con otro umbral las fechas de cada fase cambiarían. La fase más reciente queda " +
  "marcada 'en curso' porque todavía no se confirmó el próximo cruce de 20% en sentido opuesto.";

export const DRAWDOWN_VS_PHASES_NOTE =
  "Fase 26: los números de esta tabla y los de Drawdowns (más arriba) pueden describir la MISMA caída y " +
  "mostrar porcentajes distintos sin contradecirse — no es un error. El drawdown se mide siempre desde el " +
  "MÁXIMO HISTÓRICO de toda la serie; una fase bajista se mide desde el techo de ESA tendencia puntual, que " +
  "puede ser más bajo que el máximo histórico. Miden el mismo tramo del gráfico desde puntos de partida " +
  "distintos.";

export const HALVING_CYCLE_HELP =
  "El halving reduce a la mitad la recompensa por bloque de Bitcoin cada ~4 años (hecho conocido, no un " +
  "ajuste estadístico). La narrativa de un 'ciclo de 4 años' post-halving es popular, pero Bitcoin solo tuvo " +
  "4 halvings en TODA su historia — n=4 es una muestra estadísticamente insuficiente para confirmar un " +
  "patrón, por más conocida que sea la narrativa. Tratá cada ciclo como UN dato entre 4, no como una regla.";

export const MONTHLY_HEATMAP_HELP =
  "Retorno compuesto de cada mes, año por año — a diferencia de un promedio único por mes (que mezcla todos " +
  "los años en un solo número y esconde qué tan distinto fue cada año), acá se ve la estacionalidad REAL: " +
  "si un mes 'suele' ser positivo, pero con años muy dispares, el patrón es mucho más débil de lo que " +
  "sugeriría el promedio solo. Verde = mes positivo, rojo = negativo; casillero vacío = sin datos ese mes-año.";

export const STATS_SYNTHESIS_TEXT =
  "Todos estos análisis apuntan a lo mismo: la DIRECCIÓN del precio no muestra estructura predecible (el " +
  "precio no es estacionario según ADF, la autocorrelación de retornos ronda 0, la estacionalidad semanal es " +
  "ruido), pero la VOLATILIDAD y el RIESGO sí (clustering de volatilidad en la ACF de retornos², regímenes " +
  "que se sostienen en el tiempo, drawdowns que se repiten con formas parecidas). Por eso esta herramienta se " +
  "enfoca en gestionar riesgo (ver la pestaña Riesgo), no en predecir dirección.";

// --------------------------------------------------------------------------
// Vista "Comparación" (Fase 12a) — /api/compare
// --------------------------------------------------------------------------

export const COMPARISON_INTRO_HELP =
  "Comparación de rendimiento histórico normalizado: todas las monedas elegidas arrancan en 100 en la " +
  "primera fecha en que TODAS tienen dato dentro del período elegido, y desde ahí se ve cuánto creció cada " +
  "una en términos relativos. El desempeño pasado NO predice el futuro.";

export const COMPARISON_LOG_SCALE_HELP =
  "Fase 27: en escala LINEAL, una moneda que subió mucho más que las demás (por ejemplo +2000% contra " +
  "+400%) aplasta visualmente a las otras — todas las líneas de abajo se ven casi planas en comparación. En " +
  "escala LOGARÍTMICA la misma variación porcentual ocupa el mismo espacio vertical en cualquier nivel, así " +
  "que se puede comparar el comportamiento de todas las monedas aunque una haya subido muchas veces más que " +
  "otra. Empieza activada por default porque es la que realmente sirve para comparar.";

export const COMPARISON_RISK_HONEST_TEXT =
  "Rendimiento alto suele venir acompañado de riesgo alto — no es una coincidencia, es la relación más " +
  "básica de las finanzas. 'Quién subió más' (rendimiento) y 'quién rindió mejor ajustado por riesgo' " +
  "(Sharpe) no son la misma pregunta, y pueden dar respuestas distintas: una moneda puede liderar el " +
  "ranking de rendimiento y quedar peor posicionada en volatilidad, drawdown o Sharpe. Mirá las cuatro " +
  "columnas de la tabla antes de sacar conclusiones sobre cuál 'rindió mejor' — coherente con el resto de " +
  "esta herramienta, que se enfoca en gestionar riesgo, no en perseguir el rendimiento más alto.";

export const COMPARISON_RANKING_HELP =
  "Cuatro métricas del MISMO período elegido, no solo rendimiento: rendimiento total (dónde termina cada " +
  "línea del gráfico), volatilidad anualizada (cuánto se movió el precio en el camino), máximo drawdown " +
  "(la peor caída que tuviste que aguantar) y Sharpe (retorno ajustado por volatilidad). Hacé clic en el " +
  "encabezado de una columna para ordenar por esa métrica — no es un ranking de calidad del activo, solo de " +
  "cómo le fue en ESTE período puntual.";

// --------------------------------------------------------------------------
// Vista "Arbitraje" (Fase 12b) — /api/pairs/screening, /api/pairs/detail.
// ENCUADRE HONESTO: esto es arbitraje ESTADÍSTICO (pairs trading), NO
// arbitraje entre exchanges (no hay diferencia de precio del mismo activo
// entre dos mercados acá). La Fase 2 de este proyecto ya mostró que la
// mayoría de los pares de config.UNIVERSE NO están establemente
// cointegrados — esta vista muestra ese resultado tal cual, sin maquillar.
// --------------------------------------------------------------------------

export const ARBITRAGE_PURPOSE_HEADER =
  "Esta sección aplica arbitraje estadístico (pairs trading) con rigor a las 5 monedas del proyecto. Su " +
  "resultado es NEGATIVO: ningún par es operable de forma confiable. Mostramos el análisis completo no para " +
  "encontrar oportunidades, sino para DESCARTARLAS con evidencia — porque descartar una estrategia con " +
  "rigor es tan valioso como encontrar una que funcione. Es la misma honestidad que rige toda la " +
  "herramienta (ver también la pestaña Research).";

export const ARBITRAGE_INTRO_HELP =
  "Esto es arbitraje ESTADÍSTICO ('pairs trading'), no arbitraje entre exchanges: no se compra y vende el " +
  "mismo activo en dos mercados, se busca un PAR de monedas cuyos precios se mueven juntos de forma estable, " +
  "para apostar a que un desvío temporal entre ellas revierte. Con las 5 monedas de este proyecto, la mayoría " +
  "de los pares NO cumplen esa condición de forma estable — el ranking de abajo lo muestra tal cual.";

export const ARBITRAGE_SCREENING_HELP =
  "Por cada par se re-testea cointegración en ventanas móviles de ~1 año (en vez de una sola vez sobre toda " +
  "la historia) y se mide en qué fracción de esas ventanas el par siguió cointegrado. 'Estable/operable' " +
  "(verde) requiere que esa fracción sea de al menos 60% — un par cointegrado en una sola foto histórica, " +
  "pero no de forma consistente en el tiempo, se marca en rojo: no hay evidencia de que la relación se " +
  "sostenga hacia adelante. Siempre calculado sobre velas DIARIAS, sin importar el timeframe elegido arriba.";

export const ARBITRAGE_CONCEPTS_HELP = {
  cointegracion:
    "Dos activos están cointegrados si, aunque cada uno individualmente se mueva como un 'paseo aleatorio' " +
    "(sin nivel fijo), existe una combinación lineal de ambos (el spread) que sí es estable y vuelve a su " +
    "media. Es la condición de base para que un spread entre dos monedas sea operable — sin ella, no hay " +
    "ancla que garantice que un desvío revierta.",
  spread:
    "Diferencia entre el log-precio de la moneda Y y beta veces el log-precio de la moneda X (el residuo de " +
    "la regresión que estima 'beta', el hedge ratio). Si el par está genuinamente cointegrado, este spread " +
    "oscila alrededor de un nivel estable en vez de irse a cualquier lado sin límite.",
  zscore:
    "Qué tan lejos está el spread de HOY respecto de su propio promedio histórico, medido en desvíos " +
    "estándar. z=0 es el promedio; |z|>2 se considera una zona extrema. Un z-score alto NO es una señal " +
    "confiable de por sí — solo tiene sentido si el par de abajo está establemente cointegrado (ver el " +
    "veredicto de estabilidad).",
  halfLife:
    "Vida media de reversión: cuántos períodos (días u horas, según el timeframe elegido) tardaría, en " +
    "promedio, un desvío del spread en reducirse a la mitad — asumiendo que el spread efectivamente revierte. " +
    "'Sin dato' significa que, en la muestra usada, el spread no mostró reversión a la media (podría ser un " +
    "paseo aleatorio disfrazado de spread).",
  estabilidad:
    "Fracción de ventanas móviles de ~1 año en las que el par siguió cointegrado. Un p-valor bajo calculado " +
    "sobre TODA la historia de una sola vez (como en la tarjeta '¿cointegrado?') puede esconder que la " +
    "relación solo se sostuvo en un tramo puntual — esta fracción es el filtro más honesto de si de verdad " +
    "conviene operar el spread.",
};

// --------------------------------------------------------------------------
// Vista "Correlación" (Fase 13b) — /api/correlation
// --------------------------------------------------------------------------

export const CORRELATION_INTRO_HELP =
  "Correlación entre los RETORNOS diarios (u horarios) de cada par de monedas — no entre sus precios: " +
  "dos precios pueden parecer muy correlacionados solo porque ambos vienen subiendo con el tiempo, sin " +
  "que eso diga nada sobre cómo se mueven día a día. Correlación cercana a +1 = se mueven casi juntas " +
  "(poca diversificación real entre ellas); cercana a 0 = se mueven independientes; cercana a -1 = se " +
  "mueven en sentido opuesto. Las correlaciones NO son fijas — cambian con el tiempo y con el período " +
  "elegido, esto es una foto del rango de arriba, no una constante del mercado. El gráfico de más abajo " +
  "muestra justamente ESO: cómo cambia en el tiempo.";

export const CORRELATION_ROLLING_HELP =
  "Correlación de Pearson recalculada en una ventana móvil (90 días por defecto) entre las dos monedas " +
  "elegidas, día a día — a diferencia del mapa de calor de arriba (un único número sobre todo el " +
  "período), esto muestra CÓMO cambió esa relación con el tiempo. Cuando la línea se dispara hacia 1, la " +
  "diversificación entre estas dos monedas desaparece — suele pasar justo en las crisis, cuando más se " +
  "necesitaría que los activos se muevan distinto entre sí.";

export const CORRELATION_DIVERSIFICATION_INTRO =
  "Traduciendo la matriz de arriba: con todas las correlaciones entre las 5 monedas moviéndose en un " +
  "rango relativamente alto, este universo ofrece POCA diversificación real — cuando cae una moneda, " +
  "tienden a caer todas juntas. Tener las 5 en cartera no reduce el riesgo tanto como parecería a simple " +
  "vista contar 5 activos distintos. Es coherente con el enfoque de este proyecto: la gestión de riesgo " +
  "(vol targeting, ver la pestaña Riesgo) importa más que la ilusión de diversificar entre criptoactivos " +
  "que, en la práctica, se mueven juntos la mayor parte del tiempo.";

export const CORRELATION_DIVERSIFICATION_INDEX_HELP =
  "1 menos la correlación promedio entre todos los pares de la matriz (excluyendo la diagonal, que " +
  "siempre es 1). Un número cercano a 0 indica que el universo se mueve casi como un solo activo — poca " +
  "diversificación real entre las monedas elegidas; cercano a 1 indicaría activos genuinamente " +
  "independientes entre sí. No es una métrica estándar de la industria, es un resumen de lectura rápida " +
  "de la misma matriz que ya se ve arriba.";

export const CORRELATION_VS_HISTORICAL_HELP =
  "Compara la correlación 'actual' (el último valor de la ventana móvil) contra el promedio de esa misma " +
  "correlación rolling a lo largo de TODA la historia común disponible entre las dos monedas — no solo el " +
  "período que se está graficando. Por encima del promedio histórico sugiere que las dos monedas se están " +
  "moviendo MÁS juntas que lo habitual (una señal típica de estrés de mercado); por debajo, que se están " +
  "moviendo más independientes que de costumbre.";

// --------------------------------------------------------------------------
// Panel de alertas (Fase 8d, ampliado en Fase 13c)
// --------------------------------------------------------------------------

export const ALERTS_HONESTY_HELP =
  "Estas alertas son TÉCNICAS, no predicen nada: son solo el aviso de que una condición que VOS elegiste " +
  "(un cruce de precio, un nivel de RSI, tocar el POC, etc.) se cumplió en los datos más recientes. No hay " +
  "notificaciones push ni por email — solo funcionan con esta página abierta en el navegador. Una regla " +
  "creada para una moneda distinta a la que estás mirando queda guardada pero no se evalúa hasta que " +
  "cambies a verla (mirá el badge 'en vivo' de cada regla).";

export const CORRELATION_METHOD_HELP =
  "Pearson mide relación LINEAL: qué tan bien un movimiento se explica como múltiplo constante del otro. " +
  "Spearman mide relación de RANGOS (monótona): si uno sube, ¿el otro tiende a subir también, aunque no " +
  "en la misma proporción? Spearman es más robusto a valores extremos (frecuentes en retornos cripto) y " +
  "no asume que la relación sea una línea recta — con series muy volátiles puede dar un número algo " +
  "distinto a Pearson sobre los mismos datos, ninguno de los dos es 'el correcto' de forma universal.";

export const ARBITRAGE_NOT_OPERABLE_WARNING =
  "Este par NO está establemente cointegrado (menos del 60% de las ventanas móviles lo confirman). El " +
  "z-score de abajo puede igual mostrarse 'extremo' en este momento, pero sin cointegración estable ESO NO " +
  "ES UNA SEÑAL CONFIABLE: no hay garantía estadística de que el spread vaya a revertir. Tratá este análisis " +
  "como una exploración, no como una recomendación para operar.";

// Fase 15b: scatter + regresión, y backtest de la estrategia sobre el spread.

export const ARBITRAGE_ZSCORE_EXTREMES_HELP =
  "Los puntos marcados son los momentos históricos donde el z-score tocó ±2 (zona extrema). Mirar si esos " +
  "extremos volvieron hacia 0 después (revirtieron) o si el spread se quedó estirado es una forma visual de " +
  "chequear si la reversión es real o solo pasó una vez por casualidad.";

export const ARBITRAGE_SCATTER_HELP =
  "Cada punto es un día: log-precio de X en el eje horizontal, log-precio de Y en el vertical. La línea es " +
  "la recta de regresión (beta, la misma que arma el spread) — si los puntos están bien pegados a la línea, " +
  "la relación lineal entre las dos monedas es fuerte; si están dispersos y desparramados, es débil, aunque " +
  "el spread pueda parecer razonable en otras métricas.";

export const ARBITRAGE_SCATTER_NOISE_NOTE =
  "Cada punto es un día (precio de una moneda vs. la otra, en log). Que la nube siga la recta significa que " +
  "se mueven juntas; pero la DISPERSIÓN alrededor de la recta (lo 'gorda' que es la nube) es el ruido que " +
  "hace que el spread no revierta de forma confiable — por eso un par puede moverse junto la mayor parte del " +
  "tiempo y aun así no ser operable.";

export const ARBITRAGE_ZSCORE_NOT_ACTIONABLE_TEXT =
  "Un z-score extremo (>2 o <-2) sería una señal de entrada SI este par estuviera establemente cointegrado. " +
  "Como NO lo está (menos del 60% de las ventanas móviles lo confirman), este z-score NO es accionable — es " +
  "solo dónde está el spread ahora mismo, sin ninguna garantía de que vaya a volver hacia su media.";

export const ARBITRAGE_ZSCORE_ACTIONABLE_TEXT =
  "Este par SÍ está establemente cointegrado, así que un z-score extremo (>2 o <-2) acá tiene una base " +
  "estadística real como señal de entrada — mirá el backtest de más abajo para ver cómo se comportó esa " +
  "señal en la práctica, con costos incluidos.";

export const ARBITRAGE_PAIR_BACKTEST_HELP =
  "Simula la estrategia de reversión sobre ESTE par: entra corto-spread si el z-score sube por encima del " +
  "umbral de entrada, largo-spread si baja por debajo, cierra cuando vuelve cerca de la media, y corta la " +
  "pérdida (stop) si se sigue alejando. Dollar-neutral, rebalanceado a diario, con costos de transacción — " +
  "ver el detalle de los supuestos en el backend (pairs/backtest.py). Es un backtest, no un simulador de lo " +
  "que pasaría operando en vivo.";

export const ARBITRAGE_PAIR_BACKTEST_NOT_OPERABLE_WARNING =
  "Este backtest corre la estrategia igual, mecánicamente, sin importar si el par es operable o no — y eso " +
  "es a propósito: sobre un par que NO está establemente cointegrado, lo ESPERABLE es que el backtest " +
  "muestre pérdidas, un Sharpe pobre, o un resultado que no se sostiene. Verlo acá es para CONFIRMAR " +
  "cuantitativamente que el par no es operable, no una sugerencia de que valga la pena operarlo igual.";

export const ARBITRAGE_PAIR_BACKTEST_FLAT_PERIODS_NOTE =
  "Los tramos planos en la curva de equity son períodos SIN operaciones (el z-score del spread no cruzó los " +
  "umbrales de entrada/salida) — es el comportamiento correcto de la estrategia, no un cuelgue ni un error de " +
  "datos: mientras el spread se mueve cerca de su media, no hay señal para entrar.";

// --------------------------------------------------------------------------
// Vista "Research" (Fase 8c, rehecha en Fase 24) — /api/prediction (ML
// supervisado, on-demand) y /api/research-experiments (RL y rotación,
// resultados YA guardados, se leen tal cual). Toda esta vista es
// investigación con resultado NEGATIVO: ningún enfoque encontró una ventaja
// operable. Ese resultado ES el hallazgo, no un defecto a disimular.
// --------------------------------------------------------------------------

export const RESEARCH_THESIS_TEXT =
  "Acá documentamos todo lo que probamos para PREDECIR el mercado, con validación rigurosa. El resultado, " +
  "consistente en todos los enfoques, es que la dirección de estos criptoactivos no se predice de forma " +
  "confiable con datos y métodos accesibles. Eso NO es un fracaso: es un hallazgo, y es lo que evita que " +
  "esta herramienta te venda humo. La honestidad de esta sección es el diferencial de todo el proyecto: si " +
  "algo funcionara de verdad para predecir dirección, no estaría documentado gratis acá.";

export const RESEARCH_ML_APPROACH_HELP =
  "Gradient boosting (XGBoost) sobre indicadores técnicos (y features on-chain para BTC/ETH) para predecir la " +
  "dirección de la próxima vela. Etiquetas por triple-barrera (López de Prado, no un simple 'subió/bajó a N " +
  "días') y validación cruzada PURGADA con embargo: los folds de entrenamiento nunca incluyen datos que se " +
  "solapan en el tiempo con lo que se evalúa, la fuga de información más común en ML financiero. Corré la " +
  "predicción vos mismo más abajo para ver el resultado actualizado.";

export const RESEARCH_RL_APPROACH_HELP =
  "Un agente de Reinforcement Learning (PPO) decide qué fracción de la cartera poner en cada moneda (o en " +
  "cash) para maximizar el retorno ajustado por riesgo — a diferencia del ML supervisado, no predice una " +
  "clase, aprende una política de asignación directamente. Se evalúa con walk-forward (entrena solo con el " +
  "pasado, el tramo OOS siempre es futuro respecto del entrenamiento) y varias semillas aleatorias, porque un solo " +
  "entrenamiento de una red neuronal puede salir bien o mal por azar de la inicialización — una corrida sola " +
  "no prueba nada.";

export const RESEARCH_ROTATION_APPROACH_HELP =
  "El test más simple posible antes de complicarlo con ML: rotar entre dos monedas relacionadas hacia la que " +
  "tuvo mejor momentum relativo reciente, en vez de mantener las dos fijas. La lógica es la misma que motivó " +
  "el resto de la investigación — si una regla tan simple no le gana de forma robusta a sus baselines después " +
  "de costos, es poco probable que un modelo más complejo la rescate.";

export const RESEARCH_RL_TABLE_HELP =
  "Cada fila es una estrategia; 'media ± std' es el Sharpe OOS promediado sobre las semillas de la red (0 para " +
  "los baselines determinísticos, que no dependen de una semilla). El veredicto exige que la PEOR semilla del " +
  "agente — no el promedio — supere a TODOS los baselines: alcanza con una sola corrida mala para responder " +
  "que no hay ventaja consistente, porque en la práctica no sabrías de antemano con qué semilla te va a tocar " +
  "entrenar el modelo real.";

export const RESEARCH_ROTATION_TABLE_HELP =
  "Se corrieron todas las combinaciones de lookback (ventana de momentum) y frecuencia de rebalanceo para cada " +
  "par de monedas — sin quedarse con la mejor combinación después de verla (eso sería sobreajustar el reporte, " +
  "no la estrategia). 'Robusto' exige ganarle a su mejor baseline en TODAS las combinaciones del par, no en " +
  "una elegida a mano. Que solo 1 de 10 pares sea robusto es consistente con lo que esperarías por puro azar " +
  "probando 10 pares al azar, no con un patrón real de momentum explotable.";

export const RESEARCH_SYNTHESIS_TEXT =
  "Múltiples enfoques independientes — indicadores técnicos, ML supervisado con y sin datos on-chain, datos " +
  "horarios, arbitraje estadístico entre pares, Deep RL y rotación por momentum — todos validados con el " +
  "mismo rigor (sin fugas, contra baselines triviales, sin quedarse con el mejor resultado después de verlo), " +
  "llegan a la misma conclusión: ninguno encuentra una ventaja predictiva consistente sobre estos " +
  "criptoactivos. Lo único que funcionó de forma consistente en este proyecto es la GESTIÓN de riesgo (vol " +
  "targeting, ver la pestaña Riesgo) — no la predicción de dirección. Ese patrón, repetido en enfoques tan " +
  "distintos entre sí, es lo que hace confiable al resto de la herramienta: no promete lo que no puede cumplir.";

export const RESEARCH_METRIC_HELP: Record<string, string> = {
  accuracy_media: "Fracción de predicciones OOS correctas (LONG/FLAT/SHORT), promediada sobre los folds purgados. Con 3 clases, el azar puro ronda 33% — un modelo real necesita superar eso con margen y de forma consistente, no en un fold suelto.",
  baseline_azar: "Accuracy esperada de un clasificador que elige al azar entre las clases, respetando su frecuencia real (no necesariamente 33% exacto si las clases están desbalanceadas). La vara más baja que cualquier modelo debería superar.",
  baseline_mayoritaria: "Accuracy de predecir SIEMPRE la clase más frecuente del período, sin mirar ningún dato de entrada. Una vara más exigente que el azar puro, y la que más modelos de ML mal evaluados no logran superar.",
  roc_auc_media: "Área bajo la curva ROC (one-vs-rest, promediada entre clases): qué tan bien el modelo separa una clase de las demás en TODOS los umbrales posibles, no solo en el que se usó para decidir. 0.5 = azar, 1.0 = separación perfecta; 0.53 está prácticamente en 0.5.",
};
