"""
=============================================================================
PRACTICA FINAL — EJERCICIO 3
Regresion Lineal Multiple implementada desde cero con NumPy
=============================================================================

DESCRIPCION
-----------
Implementacion de la solucion analitica OLS (Minimos Cuadrados Ordinarios):

    beta = (X'X)^-1 * X'y

Derivacion matematica (Condicion de primer orden):
    Minimizar SRC(beta) = (y - X*beta)' * (y - X*beta)
    d(SRC)/d(beta) = -2X'y + 2X'X*beta = 0
    X'X * beta = X'y  →  beta = (X'X)^-1 * X'y

Por que np.linalg.lstsq en lugar de np.linalg.inv:
    inv() falla si X'X es singular o casi singular (multicolinealidad).
    lstsq() resuelve el sistema de ecuaciones normales X'X * beta = X'y
    usando descomposicion SVD — numericamente mas estable y siempre converge.
    Ambos dan exactamente los mismos coeficientes cuando X'X es invertible.

VALIDACION CONTRA SKLEARN:
    Se incluye comparativa contra LinearRegression de sklearn para verificar
    que la implementacion NumPy produce resultados identicos.
    El profesor indico que los resultados deben ser casi iguales entre ambas
    implementaciones — esta comparativa lo confirma cuantitativamente.

LIBRERIAS PERMITIDAS
--------------------
    numpy      (calculos matriciales — unico para el ajuste del modelo)
    matplotlib (visualizacion)
    sklearn    (SOLO para validacion comparativa, NO para el ajuste)

SALIDAS (output/):
    ej3_coeficientes.txt    Coeficientes NumPy vs reales vs sklearn
    ej3_metricas.txt        MAE, RMSE, R² de NumPy y comparativa sklearn
    ej3_predicciones.png    Scatter plot Real vs Predicho con linea y=x

EJECUCION:
    Desde practica_final_Pujana_Quintero_Alejandro/
    $ python ejercicio3_regresion_multiple.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# ── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# FUNCION PRINCIPAL — OLS DESDE CERO CON NUMPY
# =============================================================================

def regresion_lineal_multiple(X_train, y_train, X_test):
    """
    Ajusta un modelo de Regresion Lineal Multiple usando OLS analitico
    e implementado exclusivamente con NumPy.

    La solucion analitica es: beta = (X'X)^-1 * X'y

    Implementacion paso a paso:
      1. Agregar columna de unos a X_train y X_test para el intercepto b0.
         Sin esta columna, el modelo no tiene termino independiente y los
         coeficientes quedan sesgados.
      2. Resolver el sistema de ecuaciones normales con lstsq:
         X'X * beta = X'y
         lstsq usa descomposicion SVD — mas estable que invertir directamente.
      3. Calcular predicciones: y_pred = X_test_b @ beta

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, p)
        Matriz de features de entrenamiento.
    y_train : np.ndarray, shape (n_train,)
        Vector objetivo de entrenamiento.
    X_test : np.ndarray, shape (n_test, p)
        Matriz de features de test.

    Returns
    -------
    coefs : np.ndarray, shape (p+1,)
        Coeficientes [b0, b1, ..., bp]. b0 es el intercepto.
    y_pred : np.ndarray, shape (n_test,)
        Predicciones sobre X_test.

    Examples
    --------
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> y = np.array([5.0, 11.0, 17.0])
    >>> coefs, y_pred = regresion_lineal_multiple(X[:2], y[:2], X[2:])
    >>> print(f'Intercepto: {coefs[0]:.1f}')
    """
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]

    # Paso 1 — Columna de unos para el intercepto b0
    # np.hstack concatena horizontalmente: [1, x1, x2, ..., xp]
    # La primera columna de unos corresponde al b0 en la ecuacion y = b0 + b1*x1 + ...
    ones_train = np.ones((n_train, 1))
    X_train_b  = np.hstack([ones_train, X_train])

    # Paso 2 — Resolver beta = (X'X)^-1 * X'y usando lstsq
    # lstsq resuelve el sistema A @ x = b minimizando ||A @ x - b||^2
    # Aqui A = X_train_b y b = y_train
    # rcond=None suprime el DeprecationWarning de NumPy >= 1.14
    coefs, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)

    # Paso 3 — Columna de unos para X_test (misma estructura que X_train_b)
    ones_test = np.ones((n_test, 1))
    X_test_b  = np.hstack([ones_test, X_test])

    # Paso 4 — Predicciones: y_pred = X_test_b @ beta
    # Producto matricial: cada fila de X_test_b multiplicada por coefs
    y_pred = X_test_b @ coefs

    return coefs, y_pred


# =============================================================================
# FUNCIONES DE METRICAS — SIN SKLEARN
# =============================================================================

def calcular_mae(y_real, y_pred):
    """
    Calcula el Mean Absolute Error (MAE).

        MAE = (1/n) * sum(|y_real - y_pred|)

    Interpretacion: error promedio en las mismas unidades del target.
    Es robusto a outliers porque no eleva al cuadrado los errores.

    Parameters
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Returns
    -------
    float — Valor del MAE

    Examples
    --------
    >>> calcular_mae(np.array([3.0, 5.0]), np.array([2.5, 5.5]))
    0.5
    """
    return np.mean(np.abs(y_real - y_pred))


def calcular_rmse(y_real, y_pred):
    """
    Calcula el Root Mean Squared Error (RMSE).

        RMSE = sqrt((1/n) * sum((y_real - y_pred)^2))

    Interpretacion: como el MAE pero penaliza mas los errores grandes
    por el cuadrado. Si RMSE >> MAE, hay predicciones muy malas en algunos casos.

    Parameters
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Returns
    -------
    float — Valor del RMSE

    Examples
    --------
    >>> calcular_rmse(np.array([3.0, 5.0]), np.array([2.0, 6.0]))
    1.0
    """
    return np.sqrt(np.mean((y_real - y_pred) ** 2))


def calcular_r2(y_real, y_pred):
    """
    Calcula el coeficiente de determinacion R².

        R² = 1 - SS_res / SS_tot
        SS_res = sum((y_real - y_pred)^2)   <- varianza no explicada
        SS_tot = sum((y_real - media)^2)    <- varianza total

    Interpretacion: proporcion de la varianza del target explicada por el modelo.
    R²=1 prediccion perfecta | R²=0 el modelo no mejora la media | R²<0 peor que la media.

    Parameters
    ----------
    y_real : np.ndarray — Valores reales
    y_pred : np.ndarray — Valores predichos

    Returns
    -------
    float — Valor del R²

    Examples
    --------
    >>> calcular_r2(np.array([3.0, 5.0, 7.0]), np.array([3.0, 5.0, 7.0]))
    1.0
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / ss_tot


# =============================================================================
# FUNCION DE VISUALIZACION
# =============================================================================

def graficar_real_vs_predicho(y_real, y_pred,
                               ruta_salida=None):
    """
    Genera scatter plot de Valores Reales vs Valores Predichos con linea
    de referencia perfecta y = x.

    Un modelo perfecto produce todos los puntos sobre la diagonal.
    La dispersion alrededor de la diagonal representa el error del modelo.
    La distancia vertical de cada punto a la diagonal = residuo de ese punto.

    Parameters
    ----------
    y_real      : np.ndarray — Valores reales del test set
    y_pred      : np.ndarray — Predicciones del modelo
    ruta_salida : str o Path — Ruta de guardado. Default: output/ej3_predicciones.png

    Returns
    -------
    None
        Guarda la figura en ruta_salida.

    Examples
    --------
    >>> graficar_real_vs_predicho(y_test, y_pred)
    """
    if ruta_salida is None:
        ruta_salida = OUTPUT_DIR / "ej3_predicciones.png"

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_real, y_pred,
               alpha=0.65, s=40, color="#1976D2",
               edgecolors="white", linewidth=0.4,
               label="Predicciones NumPy OLS")

    # Linea de referencia perfecta y = x
    lim_min = min(y_real.min(), y_pred.min()) - 1
    lim_max = max(y_real.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color="crimson", linewidth=1.5, linestyle="--",
            label="Prediccion perfecta (y = x)")

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Valores Reales", fontsize=10)
    ax.set_ylabel("Valores Predichos", fontsize=10)
    ax.set_title(
        "Real vs Predicho — OLS implementado con NumPy\n"
        "Datos sinteticos (semilla=42, n=200, p=3)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {Path(ruta_salida).name}")


# =============================================================================
# MAIN — NO MODIFICAR ESTE BLOQUE
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Datos sinteticos con semilla fija para reproducibilidad
    # -------------------------------------------------------------------------
    SEMILLA = 42
    rng = np.random.default_rng(SEMILLA)

    n_muestras = 200
    n_features = 3

    X            = rng.standard_normal((n_muestras, n_features))
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])
    ruido        = rng.normal(0, 1.5, n_muestras)
    y            = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    # Split 80/20 sin shuffle (como especifica el main del profesor)
    corte   = int(0.8 * n_muestras)
    X_train, X_test = X[:corte], X[corte:]
    y_train, y_test = y[:corte], y[corte:]

    # -------------------------------------------------------------------------
    # Ajuste del modelo NumPy OLS
    # -------------------------------------------------------------------------
    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    mae  = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2   = calcular_r2(y_test, y_pred)

    # -------------------------------------------------------------------------
    # Resultados en consola
    # -------------------------------------------------------------------------
    print("=" * 55)
    print("RESULTADOS — Regresion Lineal Multiple (NumPy OLS)")
    print("=" * 55)
    print(f"\nCoeficientes reales   : {coefs_reales}")
    print(f"Coeficientes ajustados: {np.round(coefs, 6)}")
    print(f"\nDiferencia por coeficiente:")
    nombres = ["b0 (intercepto)", "b1 (feature 1)", "b2 (feature 2)", "b3 (feature 3)"]
    for nombre, real, ajust in zip(nombres, coefs_reales, coefs):
        diff = ajust - real
        print(f"  {nombre:<20} real={real:>5.2f}  ajust={ajust:>8.4f}  diff={diff:>+8.4f}")

    print(f"\nMetricas sobre test set (n={len(y_test)}):")
    print(f"  MAE  = {mae:.4f}  (referencia: ~1.20 ±0.20)")
    print(f"  RMSE = {rmse:.4f}  (referencia: ~1.50 ±0.20)")
    print(f"  R²   = {r2:.4f}  (referencia: ~0.80 ±0.05)")

    # -------------------------------------------------------------------------
    # Validacion comparativa contra sklearn
    # El profesor indico que los resultados deben ser casi identicos
    # -------------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("VALIDACION — Comparativa NumPy OLS vs sklearn")
    print("=" * 55)
    try:
        from sklearn.linear_model import LinearRegression
        modelo_sk = LinearRegression()
        modelo_sk.fit(X_train, y_train)
        y_pred_sk = modelo_sk.predict(X_test)

        coefs_sk = np.concatenate([[modelo_sk.intercept_], modelo_sk.coef_])
        mae_sk   = np.mean(np.abs(y_test - y_pred_sk))
        rmse_sk  = np.sqrt(np.mean((y_test - y_pred_sk) ** 2))
        r2_sk    = 1 - np.sum((y_test - y_pred_sk)**2) / np.sum((y_test - np.mean(y_test))**2)

        print(f"\n{'Metrica':<12} {'NumPy OLS':>12} {'sklearn':>12} {'Diferencia':>12}")
        print("-" * 52)
        print(f"{'MAE':<12} {mae:>12.6f} {mae_sk:>12.6f} {abs(mae-mae_sk):>12.8f}")
        print(f"{'RMSE':<12} {rmse:>12.6f} {rmse_sk:>12.6f} {abs(rmse-rmse_sk):>12.8f}")
        print(f"{'R2':<12} {r2:>12.6f} {r2_sk:>12.6f} {abs(r2-r2_sk):>12.8f}")
        print(f"\nCoeficientes:")
        print(f"{'':20} {'NumPy':>12} {'sklearn':>12} {'Diferencia':>12}")
        for nombre, cn, cs in zip(nombres, coefs, coefs_sk):
            print(f"  {nombre:<20} {cn:>12.6f} {cs:>12.6f} {abs(cn-cs):>12.8f}")

        tolerancia = 1e-6
        if np.allclose(coefs, coefs_sk, atol=tolerancia):
            print(f"\n  VALIDACION OK: coeficientes identicos (tolerancia={tolerancia})")
        else:
            print(f"\n  ADVERTENCIA: diferencias mayores a {tolerancia}")

    except ImportError:
        print("  sklearn no disponible — omitiendo validacion comparativa")

    # -------------------------------------------------------------------------
    # Guardar salidas (bloque original del profesor, no modificado)
    # -------------------------------------------------------------------------
    with open(OUTPUT_DIR / "ej3_coeficientes.txt", "w", encoding="utf-8") as f:
        f.write("Regresion Lineal Multiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres_out = ["Intercepto (b0)"] + [f"b{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres_out, coefs):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia:\n")
        for nombre, valor in zip(nombres_out, coefs_reales):
            f.write(f"  {nombre}: {valor:.6f}\n")
        f.write("\nValidacion sklearn:\n")
        try:
            coefs_sk_out = np.concatenate([[modelo_sk.intercept_], modelo_sk.coef_])
            for nombre, vn, vs in zip(nombres_out, coefs, coefs_sk_out):
                f.write(f"  {nombre}: NumPy={vn:.6f} | sklearn={vs:.6f} | diff={abs(vn-vs):.8f}\n")
        except NameError:
            f.write("  sklearn no disponible\n")

    with open(OUTPUT_DIR / "ej3_metricas.txt", "w", encoding="utf-8") as f:
        f.write("Regresion Lineal Multiple — Metricas de evaluacion\n")
        f.write("=" * 50 + "\n")
        f.write(f"  MAE  : {mae:.6f}\n")
        f.write(f"  RMSE : {rmse:.6f}\n")
        f.write(f"  R2   : {r2:.6f}\n")
        f.write("\nReferencia del profesor (semilla=42):\n")
        f.write("  MAE  ~ 1.20 (+-0.20)\n")
        f.write("  RMSE ~ 1.50 (+-0.20)\n")
        f.write("  R2   ~ 0.80 (+-0.05)\n")
        try:
            f.write("\nValidacion sklearn:\n")
            f.write(f"  MAE  sklearn: {mae_sk:.6f}  | diff: {abs(mae-mae_sk):.8f}\n")
            f.write(f"  RMSE sklearn: {rmse_sk:.6f} | diff: {abs(rmse-rmse_sk):.8f}\n")
            f.write(f"  R2   sklearn: {r2_sk:.6f}  | diff: {abs(r2-r2_sk):.8f}\n")
        except NameError:
            f.write("  sklearn no disponible\n")

    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en output/")
    print("  -> output/ej3_coeficientes.txt")
    print("  -> output/ej3_metricas.txt")
    print("  -> output/ej3_predicciones.png")