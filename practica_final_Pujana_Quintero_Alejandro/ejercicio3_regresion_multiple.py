"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 3
Regresión Lineal Múltiple implementada desde cero con NumPy
=============================================================================

DESCRIPCIÓN
-----------
En este ejercicio debes implementar la función `regresion_lineal_multiple`
que ajusta un modelo de regresión lineal múltiple utilizando la solución
analítica de Mínimos Cuadrados Ordinarios (OLS):

    β = (XᵀX)⁻¹ Xᵀy

La función debe ser capaz de:
  1. Añadir el término independiente (intercepto) automáticamente.
  2. Calcular los coeficientes β₀, β₁, ..., βₙ.
  3. Devolver las predicciones ŷ para un conjunto de datos nuevo.
  4. Calcular las métricas de evaluación: MAE, RMSE y R².

LIBRERÍAS PERMITIDAS
--------------------
  - numpy   (cálculos matriciales)
  - matplotlib (visualización, opcional)

NO está permitido usar sklearn para el ajuste del modelo en este ejercicio.

SALIDAS ESPERADAS (carpeta output/)
------------------------------------
  - output/ej3_coeficientes.txt   → Coeficientes del modelo ajustado
  - output/ej3_metricas.txt       → MAE, RMSE y R² sobre datos de test
  - output/ej3_predicciones.png   → Gráfico Real vs. Predicho

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)


# =============================================================================
# FUNCIÓN PRINCIPAL — COMPLETA ESTA SECCIÓN
# =============================================================================

def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de Regresión Lineal Múltiple usando OLS y devuelve
    las predicciones sobre el conjunto de test.

    La solución analítica es:
        β = (XᵀX)⁻¹ Xᵀy

    Parámetros
    ----------
    X_train : np.ndarray de forma (n_train, p)
        Matriz de features de entrenamiento. Cada fila es una observación
        y cada columna es una variable predictora.
    y_train : np.ndarray de forma (n_train,)
        Vector de valores objetivo de entrenamiento.
    X_test : np.ndarray de forma (n_test, p)
        Matriz de features sobre la que se quiere predecir.

    Retorna
    -------
    coefs : np.ndarray de forma (p+1,)
        Vector de coeficientes ajustados [β₀, β₁, ..., βₚ].
        β₀ es el intercepto (término independiente).
    y_pred : np.ndarray de forma (n_test,)
        Predicciones del modelo sobre X_test.

    Pistas
    ------
    - Usa np.hstack o np.column_stack para añadir la columna de unos (intercepto).
    - Usa np.linalg.inv o np.linalg.lstsq para resolver el sistema.
    - np.linalg.lstsq es numéricamente más estable que invertir directamente.
    """

    # TODO: Paso 1 — Añadir columna de unos a X_train para el intercepto β₀
    # Pista: np.ones((n, 1)) y np.hstack([ones, X_train])
    X_train_b = None  # ← Reemplaza None con tu implementación

    # TODO: Paso 2 — Calcular los coeficientes β con la fórmula OLS
    # β = (XᵀX)⁻¹ Xᵀy
    coefs = None  # ← Reemplaza None con tu implementación

    # TODO: Paso 3 — Añadir columna de unos a X_test de la misma forma
    X_test_b = None  # ← Reemplaza None con tu implementación

    # TODO: Paso 4 — Calcular predicciones ŷ = X_test_b · β
    y_pred = None  # ← Reemplaza None con tu implementación

    return coefs, y_pred


# =============================================================================
# FUNCIONES DE MÉTRICAS — COMPLETA ESTA SECCIÓN
# =============================================================================

def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error (MAE).

        MAE = (1/n) * Σ |y_real - y_pred|

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del MAE
    """
    # TODO: Implementa el MAE sin usar sklearn
    pass


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error (RMSE).

        RMSE = sqrt((1/n) * Σ (y_real - y_pred)²)

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del RMSE
    """
    # TODO: Implementa el RMSE sin usar sklearn
    pass


def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinación R².

        R² = 1 - SS_res / SS_tot
        SS_res = Σ (y_real - y_pred)²
        SS_tot = Σ (y_real - ȳ)²

    Parámetros
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Retorna
    -------
    float — Valor del R² (entre -∞ y 1; cuanto más cercano a 1, mejor)
    """
    # TODO: Implementa el R² sin usar sklearn
    pass


# =============================================================================
# FUNCIÓN DE VISUALIZACIÓN — COMPLETA ESTA SECCIÓN (OPCIONAL)
# =============================================================================

def graficar_real_vs_predicho(y_real, y_pred, ruta_salida="output/ej3_predicciones.png"):
    """
    Genera un scatter plot de Valores Reales vs. Valores Predichos.

    Un modelo perfecto produciría todos los puntos sobre la diagonal y=x.
    La dispersión alrededor de esa línea representa el error del modelo.

    Parámetros
    ----------
    y_real      : np.ndarray — Valores reales del test set
    y_pred      : np.ndarray — Predicciones del modelo
    ruta_salida : str        — Ruta donde guardar la imagen
    """
    # TODO: Implementa la visualización
    # Pistas:
    #   - plt.scatter(y_real, y_pred, alpha=0.6)
    #   - Dibuja la línea de referencia perfecta: y = x
    #   - Añade etiquetas a los ejes y título
    #   - Guarda con plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    pass


# =============================================================================
# MAIN — NO MODIFIQUES ESTE BLOQUE (es la prueba de referencia del profesor)
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Datos sintéticos con semilla fija para reproducibilidad
    # -------------------------------------------------------------------------
    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    # Generamos features aleatorias
    X = rng.standard_normal((n_muestras, n_features))

    # Coeficientes "reales" conocidos: β₀=5, β₁=2, β₂=-1, β₃=0.5
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])

    # Variable objetivo con ruido gaussiano (σ=1.5)
    ruido = rng.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    # -------------------------------------------------------------------------
    # Split Train / Test (80% / 20%) — sin mezclar aleatoriamente
    # -------------------------------------------------------------------------
    corte = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # -------------------------------------------------------------------------
    # Ajuste del modelo
    # -------------------------------------------------------------------------
    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------
    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    # -------------------------------------------------------------------------
    # Mostrar resultados en consola
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("RESULTADOS — Regresión Lineal Múltiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:   {coefs_reales}")
    print(f"Coeficientes ajustados: {coefs}")
    print(f"\nMétricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    # -------------------------------------------------------------------------
    # RESULTADO DE REFERENCIA DEL PROFESOR
    # Con SEMILLA=42, los resultados esperados aproximados son:
    #   Coefs ajustados ≈ [5.0, 2.0, -1.0, 0.5]  (cercanos a los reales)
    #   MAE  ≈ 1.20  (±0.20 según implementación)
    #   RMSE ≈ 1.50  (±0.20 según implementación)
    #   R²   ≈ 0.80  (±0.05 según implementación)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Guardar salidas
    # -------------------------------------------------------------------------

    # Fichero de coeficientes
    with open("output/ej3_coeficientes.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (β₀)"] + [f"β{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")

    # Fichero de métricas
    with open("output/ej3_metricas.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Métricas de evaluación\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R²   : {r2:.6f}\n")

    # Gráfico
    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en la carpeta output/")
    print("  → output/ej3_coeficientes.txt")
    print("  → output/ej3_metricas.txt")
    print("  → output/ej3_predicciones.png")
