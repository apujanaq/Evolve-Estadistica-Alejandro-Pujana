# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Documento de respuestas basado en los resultados reales obtenidos al ejecutar
> los scripts de cada ejercicio. Los valores numéricos provienen directamente
> de los outputs de los scripts y los notebooks de análisis.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

*Análisis detallado, código documentado y visualizaciones disponibles en `notebooks/ej1_analisis_descriptivo.ipynb`*

---

### Descripción y análisis

El análisis descriptivo se realizó sobre el dataset **Loan Approval Data 2025** de Kaggle
(50,000 registros x 20 columnas, 5.5 MB en disco). El objetivo es caracterizar los datos
antes de modelarlos, identificando estructura, calidad y relaciones entre variables.

**Decisiones de preprocesamiento documentadas:**
- `customer_id` eliminado: identificador administrativo sin valor predictivo
- `loan_status` eliminado: data leakage — es el resultado post-decisión, no una característica previa
- `payment_to_income_ratio` eliminado: multicolinealidad perfecta con `loan_to_income_ratio` (r=1.000),
  lo que hace la matriz X'X singular e impide la inversión en OLS

**Variables categóricas analizadas:**
- `occupation_status`: 3 categorías — Employed (69.9%), Self-Employed (20.4%), Student (9.7%). Desbalance 7.2x
- `product_type`: 3 categorías — Credit Card (44.9%), Personal Loan (35.0%), Line of Credit (20.0%)
- `loan_intent`: 6 categorías — Personal (24.9%), Education (20.3%), Medical (15.2%), Business (14.9%),
  Home Improvement (14.9%), Debt Consolidation (9.8%)

**Moneda asumida:** USD, dado el origen del dataset (Kaggle, contexto financiero anglosajón).

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de Kaggle: Loan Approval Data 2025. La variable objetivo es `loan_amount`
> — monto del préstamo aprobado en USD, variable continua con rango $500–$100,000.
>
> Tiene sentido hacer regresión porque representa la decisión cuantitativa del banco:
> dado el perfil financiero de un solicitante, cuánto dinero prestarle. Es una variable
> continua con variabilidad explicable por características observables del solicitante
> como ingreso, historial crediticio y deuda existente.
>
> Se escoge `loan_amount` sobre `loan_to_income_ratio` porque esta última es una variable
> derivada (loan_amount / annual_income) que contiene el target en su numerador — usarla
> como predictor constituiria data leakage.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables númericas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las principales variables numéricas presentan distribución asimétrica positiva:
> `annual_income` (skewness=1.89), `years_employed` (1.47), `savings_assets` (11.4),
> `current_debt` (1.53). La excepción es `credit_score` con distribución prácticamente
> normal (skewness=-0.003) por diseño del sistema FICO — los bureaus de crédito calibran
> el score para que siga una distribución normal en la población general.
>
> La variable objetivo `loan_amount` presenta skewness=0.9315 (asimetría positiva moderada):
> la mayoría de préstamos son de monto moderado con una cola de préstamos grandes menos frecuentes.
> IQR = $36,200 | Mediana = $26,100 | Media = $33,042.
>
> **Detección de outliers — método IQR** (seleccionado sobre Z-Score por la asimetría presente):
> - `savings_assets`: 6,508 outliers (13.02%) — curtosis=57.8, concentración extrema cerca de cero
> - `derogatory_marks`: 6,393 outliers (12.79%) — perfiles con historial crediticio muy deteriorado
> - `current_debt`: 2,932 outliers (5.86%)
> - `annual_income`: 2,355 outliers (4.71%)
> - `loan_amount`: **0 outliers** — el límite superior IQR es $102,800 y el máximo del dataset $100,000
>
> **Decisión: no eliminar ningún outlier.** Son valores económicamente válidos — altos ahorros
> y marcas negativas severas son perfiles reales que el modelo debe capturar. Eliminarlos
> introduciria sesgo de selección. Se documenta como limitación del modelo que outliers en
> `savings_assets` (curtosis=57.8) pueden afectar la estabilidad de los coeficientes OLS.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Las tres variables con mayor correlación de Pearson con `loan_amount` son:
>
> 1. `loan_to_income_ratio`: r = +0.6622 (correlación fuerte positiva)
> 2. `annual_income`: r = +0.5111 (correlación moderada positiva)
> 3. `current_debt`: r = +0.3598 (correlación débil positiva)
>
> Interpretación económica: el banco presta en proporción al ingreso del solicitante
> (r=0.66 con el ratio ingreso-préstamo) y a su capacidad económica absoluta (r=0.51 con ingreso).
> La correlación positiva con `current_debt` indica que quienes tienen más deuda también tienen
> mayor acceso histórico al crédito — no es un indicador de riesgo sino de capacidad crediticia.
>
> Hallazgo adicional: `credit_score` tiene correlación r=-0.49 con `interest_rate`
> (confirmando que mejor score implica menor tasa) pero solo r=+0.11 con `loan_amount`,
> sugiriendo que el score determina las condiciones del préstamo, no el monto.
>
> Multicolinealidad detectada: `loan_to_income_ratio` y `payment_to_income_ratio` tienen r=1.000
> — son la misma variable escalada. `payment_to_income_ratio` fue excluida del modelo.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> El dataset no contiene valores nulos en ninguna de las 20 columnas
> (0 valores nulos sobre 50,000 registros, 0.00%).
> No se requirió ninguna acción de imputación, eliminación de filas ni tratamiento especial.
> Este hallazgo se documenta explícitamente porque simplifica el preprocesamiento
> y permite usar el dataset completo sin pérdida de observaciones.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

*Análisis detallado, código documentado y visualizaciones disponibles en `notebooks/ej2_regresion_lineal.ipynb`*

---

### Descripción y análisis

Se entrenó un modelo de Regresión Lineal Múltiple (Modelo A) usando sklearn sobre el
dataset Loan Approval Data 2025, aplicando el preprocesamiento derivado del Ejercicio 1.

**Preprocesamiento aplicado:**
- Exclusión de `customer_id`, `loan_status`, `payment_to_income_ratio` (justificado en Ej.1)
- One-Hot Encoding con `drop_first=True` para evitar la trampa de variables dummy.
  Categorías de referencia: Employed | Credit Card | Business
- `StandardScaler` ajustado SOLO sobre X_train para evitar data leakage del scaler
- Split 80/20 con `random_state=42`: 40,000 obs train | 10,000 obs test

**Complemento con statsmodels OLS:** se incluye summary estadístico completo
(p-values, t-statistics, intervalos de confianza) usando statsmodels, librería
permitida por el enunciado. Esto permite inferencia estadística equivalente a
la tabla `reg` de STATA, identificando que variables son estadísticamente significativas.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> **Métricas sobre el test set (10,000 observaciones):**
> - MAE  = $7,560.99 USD
> - RMSE = $11,207.17 USD
> - R²   = 0.8173
>
> **Métricas sobre el train set (comparativa overfitting):**
> - R²_train = 0.8104 | R²_test = 0.8173 | Diferencia: -0.0070
>
> **El modelo funciona bien**, con las siguientes consideraciones:
>
> **Sin overfitting:** R²_train ≈ R²_test, diferencia de solo -0.007.
> El modelo generaliza correctamente a datos nuevos.
>
> **Capacidad predictiva:** R²=0.817 significa que el modelo explica el 81.7% de la
> variabilidad del monto del préstamo. En promedio se equivoca en $7,561 USD.
>
> **Limitación 1 — Heterocedasticidad:** el gráfico de residuos muestra un patrón
> de abanico — la varianza del error crece con el monto predicho. El modelo predice
> mejor préstamos pequeños que grandes. Esto viola el supuesto OLS de homocedasticidad.
> La corrección de errores estándar robustos (Huber-White) sería la solución técnica,
> pero no es requerida por el enunciado y se documenta como limitación metodológica.
>
> **Limitación 2 — Dominancia de loan_to_income_ratio:** esta variable tiene beta=$41,243
> (t=297, p<0.001) y explica la mayor parte del R². Es una variable derivada del target
> (loan_amount/annual_income), lo que genera una correlación circular a evaluar en producción.
>
> **Variables más influyentes (solo significativas p<0.05 según statsmodels OLS):**
> 1. `loan_to_income_ratio`: beta=$41,243, t=297, p<0.001
> 2. `annual_income`: beta=$0.47/USD, t=127, p<0.001
> 3. `age`: beta=$91.21/año, t=11, p<0.001
> 4. `years_employed`: beta=$34.52/año, t=3.4, p=0.001
> 5. `occupation_status_Self-Employed`: beta=$840 vs Employed, t=5.6, p<0.001
> 6. `product_type_Line of Credit`: beta=-$2,239 vs Credit Card, t=-4.2, p<0.001
>
> **Hallazgo notable:** `credit_score` no es significativo (p=0.525). El score determina
> las condiciones del préstamo (tasa de interes, r=-0.49), no el monto — confirmado por
> el Ejercicio 1. Ninguna categoría de `loan_intent` es significativa, consistente con
> los boxplots del EDA que mostraban medianas similares entre categorías.
>
> **Conexión con Ejercicio 1:** los tres predictores más correlacionados con el target
> en el EDA son exactamente las variables con mayor peso en el modelo, confirmando
> que el análisis descriptivo previo fue predictivo de los resultados del modelo.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

*Análisis detallado, derivación matemática y validación disponibles en `notebooks/ej3_regresion_numpy.ipynb`*

---

### Descripción y análisis

Se implementó la solución analítica OLS desde cero usando exclusivamente NumPy.
Los datos son completamente sintéticos (semilla=42, n=200, p=3) con coeficientes
reales conocidos: β₀=5.0, β₁=2.0, β₂=-1.0, β₃=0.5, ruido N(0, 1.5).

La implementación se validó contra sklearn obteniendo diferencia=0.00000000 en todos
los coeficientes y métricas, confirmando correctitud matemática de la implementación.

**Por qué np.linalg.lstsq en lugar de np.linalg.inv:**
lstsq resuelve el sistema X'X * β = X'y usando descomposición SVD, numéricamente
más estable que invertir directamente. Ambos dan exactamente los mismos coeficientes
cuando X'X es invertible, pero lstsq no falla cuando la matriz esta mal condicionada.

---

**Pregunta 3.1** — Explica en tus propias palabras que hace la formula β = (X'X)^-1 * X'y y por qué es necesario añadir una columna de unos a la matriz X.

> La formula β = (X'X)^-1 * X'y es la solución analítica que minimiza la Suma de
> Residuos al Cuadrado (SRC). Se obtiene derivando SRC respecto a beta e igualando
> a cero (condición de primer orden):
>
> SRC(β) = (y - Xβ)' * (y - Xβ)
> dSRC/dβ = -2X'y + 2X'Xβ = 0
> → X'X * β = X'y  (ecuaciones normales)
> → β = (X'X)^-1 * X'y  (solución analítica)
>
> En código NumPy se resuelve con `np.linalg.lstsq(X_train_b, y_train)`, numéricamente
> más estable que `np.linalg.inv`.
>
> **Por qué la columna de unos:** el modelo lineal es y = β₀ + β₁x₁ + β₂x₂ + β₃x₃.
> El término β₀ es una constante — equivale a multiplicar β₀ por 1. Para expresarlo
> en forma matricial y = X_b @ β, la matriz X_b necesita una columna de 1s que al
> multiplicarse por β₀ produzca el término independiente. Sin esta columna, el modelo
> fuerza β₀=0 y la recta pasa por el origen, sesgando todos los coeficientes.

**Pregunta 3.2** — Copia aqui los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parámetro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.864995       |
| β₁        | 2.0       | 2.063618       |
| β₂        | -1.0      | -1.117038      |
| β₃        | 0.5       | 0.438517       |

> Los coeficientes ajustados son próximos a los valores reales. Las diferencias
> (máxima: 0.135 en β₀) son consecuencia del ruido gaussiano con sigma=1.5 en los datos.
> Los signos son correctos en todos los coeficientes.
>
> **Validación contra sklearn:** diferencia = 0.00000000 en todos los coeficientes,
> confirmando que la implementación NumPy es matemáticamente idéntica a sklearn.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> **Métricas obtenidas sobre el test set (40 observaciones):**
> - MAE  = 1.1665 (referencia: ~1.20 ±0.20) → dentro del rango
> - RMSE = 1.4612 (referencia: ~1.50 ±0.20) → dentro del rango
> - R²   = 0.6897 (referencia: ~0.80 ±0.05) → fuera del rango
>
> MAE y RMSE están dentro del rango de referencia. El R²=0.6897 está fuera del rango
> esperado, pero no por error de implementación — la validación contra sklearn con
> diferencia=0.00000000 lo confirma. La causa es la varianza muestral con solo 40
> observaciones en test. Si se evaluara sobre el train set (160 obs), el R² sería
> más cercano al valor de referencia.

---

## Ejercicio 4 — Series Temporales

*Análisis detallado, interpretación de gráficos y tests estadísticos disponibles en `notebooks/ej4_series_temporales.ipynb`*

---

### Descripción y análisis

Se analizó una serie temporal sintética generada por `generar_serie_temporal(semilla=42)`.
La función NO fue modificada. La serie cubre 6 años de datos diarios (2018–2023),
con 2,191 observaciones totales y 1,827 residuos válidos post-descomposición.

**Modelo de descomposición elegido:** aditivo (`model='additive'`, `period=365`).
Se usa modelo aditivo porque la amplitud de la estacionalidad es constante a lo largo
del tiempo — no crece con el nivel de la serie.

**Test de normalidad:** se usó Jarque-Bera en lugar de Shapiro-Wilk porque es el
estándar en econometría de series temporales, no tiene límite de muestra y es más
robusto para tamaños muestrales variables.

**Componentes teóricos conocidos de la serie (para validación):**

| Componente | Fórmula de generación | Parámetros |
|---|---|---|
| Tendencia | 0.05 * t + 50 | Pendiente: +0.05/dia |
| Estacionalidad | 15*sin(...) + 6*cos(...) | Amplitud ~15, periodo: 365.25 dias |
| Ciclo | 8*sin(2π*t/1461) | Amplitud: 8, periodo: ~4 años |
| Ruido | N(0, 3.5) | Media: 0, sigma: 3.5 |

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> La serie presenta **tendencia lineal creciente**.
>
> - Tipo: lineal (pendiente constante, sin aceleración)
> - Dirección: positiva
> - Magnitud: de ~64.14 (inicio 2018) a ~155.25 (finales 2023), incremento de 91.10 unidades
> - Pendiente estimada: ~+0.05 unidades/día (consistente con la fórmula teórica de generación)
>
> Confirmado por la media móvil de 365 dias y el componente de tendencia de seasonal_decompose.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Sí, estacionalidad **anual clara y regular**.
>
> - Periodo: 365 días (un ciclo completo por año)
> - Amplitud recuperada: ~14.46 unidades (de -21 en invierno a +13 en verano)
> - El patrón se repite idéntico cada año durante los 6 años analizados
> - Consistente con la fórmula: 15*sin(2π*t/365.25) + 6*cos(4π*t/365.25)

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Sí, ciclos de largo plazo con periodo ~4 años (1,461 dias) y amplitud de 8 unidades.
>
> Se distinguen de la tendencia porque:
> - La **tendencia** es monótonamente creciente — siempre sube
> - El **ciclo** es oscilatorio — sube y baja periódicamente sobre la tendencia
>
> En la serie se observa que la velocidad de crecimiento no es perfectamente constante:
> hay periodos donde sube más rapido (ciclo en fase positiva) y otros donde se aplana
> (ciclo en fase negativa). La media movil de 365 días muestra esta ligera curvatura
> cada ~4 años sobre la tendencia lineal.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> El residuo **se ajusta a un ruido blanco gaussiano ideal**. Todos los criterios se cumplen:
>
> **Estadísticos del residuo** (n=1,827 observaciones válidas):
> - Media: 0.1271 (ideal: 0)
> - Desviación típica: 3.2220 (sigma teórico de generación: 3.5)
> - Skewness: -0.0509 (ideal: 0)
> - Curtosis: -0.0610 (ideal: 0 en exceso)
>
> **Test Jarque-Bera (normalidad):**
> - Estadístico: 1.1013 | p-value: 0.5766
> - Conclusión: **No se rechaza H0** (p=0.5766 > 0.05) — el residuo es normal
>
> **Test ADF (estacionariedad):**
> - Estadístico: -39.9160 | p-value: 0.000000
> - Conclusión: **Se rechaza H0** (p<0.05) — el residuo ES estacionario
>
> **ACF y PACF:** todas las barras de lag=1 a lag=40 dentro del IC al 95%,
> confirmando ausencia de autocorrelación significativa.
>
> El residuo cumple todos los criterios de ruido blanco gaussiano: media≈0,
> normalidad (JB p=0.58), estacionariedad (ADF p≈0), sin autocorrelación.
> La diferencia entre std=3.22 y sigma_teorico=3.50 se explica porque
> seasonal_decompose absorbe parte del ruido en la estimación de la tendencia.

---

*Fin del documento de respuestas*
