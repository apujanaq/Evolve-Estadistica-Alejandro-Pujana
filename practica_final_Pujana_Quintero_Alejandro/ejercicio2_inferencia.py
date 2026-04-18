"""
=============================================================================
PRACTICA FINAL — EJERCICIO 2
Regresion Lineal con Scikit-Learn + Analisis OLS con Statsmodels
=============================================================================

DATASET : Loan Approval Data 2025 (Kaggle)
          50,000 registros x 20 columnas | 5.5 MB en disco
TARGET  : loan_amount — monto del prestamo aprobado (USD, continuo)

MODELO  : LinearRegression (sklearn) — Modelo A segun enunciado

PREPROCESAMIENTO:
  - Excluir: customer_id (ID), loan_status (data leakage),
             payment_to_income_ratio (multicolinealidad r=1.000)
  - OHE con drop_first=True para evitar trampa de variables dummy
  - StandardScaler para comparabilidad de coeficientes
  - Split 80/20 con random_state=42

COMPLEMENTO:
  - Summary OLS con statsmodels para inferencia estadistica
    (p-values, t-stats, intervalos de confianza)
  - statsmodels esta listada como libreria permitida en el enunciado

SALIDAS (output/):
  ej2_metricas_regresion.txt   MAE, RMSE, R2 train y test
  ej2_residuos.png             Grafico de residuos (predichos vs residuos)
  ej2_coeficientes.png         Top-10 coeficientes por peso absoluto
  ej2_summary_ols.txt          Tabla OLS completa con p-values y t-stats

EJECUCION:
  Desde practica_final_Pujana_Quintero_Alejandro/
  $ python ejercicio2_inferencia.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm

# ── Reproducibilidad ────────────────────────────────────────────────────────
np.random.seed(42)

# ── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "Loan_approval_data_2025.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Constantes ───────────────────────────────────────────────────────────────
TARGET    = "loan_amount"
DROP_COLS = ["customer_id", "loan_status", "payment_to_income_ratio"]
CAT_COLS  = ["occupation_status", "product_type", "loan_intent"]

sns.set_theme(style="whitegrid", palette="muted")


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocesar(df: pd.DataFrame) -> tuple:
    """
    Aplica el pipeline de preprocesamiento completo:
    1. Elimina columnas excluidas (ID, data leakage, multicolinealidad)
    2. One-Hot Encoding con drop_first=True (evita trampa de variables dummy)
    3. Split 80/20 estratificado por random_state=42
    4. StandardScaler ajustado SOLO sobre train (evita data leakage del scaler)

    Por que StandardScaler en regresion lineal:
    - Los coeficientes de sklearn son no estandarizados por defecto.
      annual_income (escala ~50,000) y credit_score (escala ~300-850) no son
      directamente comparables sin estandarizar.
    - El escalado no afecta R2 ni las predicciones finales, pero hace que el
      grafico de coeficientes refleje importancia relativa real.
    - CRITICO: el scaler se ajusta SOLO sobre X_train. Si lo ajustaras sobre
      todo el dataset antes del split, el modelo "veria" informacion del test
      set — otro tipo de data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original cargado desde CSV.

    Returns
    -------
    X_train_sc, X_test_sc : np.ndarray
        Features escaladas para train y test.
    y_train, y_test : pd.Series
        Target para train y test.
    feature_names : list[str]
        Nombres de las features post-OHE.
    scaler : StandardScaler
        Scaler ajustado (para referencia o uso futuro).
    """
    separador = "=" * 60

    print(separador)
    print("PREPROCESAMIENTO")
    print(separador)

    # 1. Eliminar columnas excluidas
    df_clean = df.drop(columns=DROP_COLS)
    print(f"  Columnas eliminadas: {DROP_COLS}")
    print(f"  Shape post-eliminacion: {df_clean.shape}")

    # 2. One-Hot Encoding
    # drop_first=True: elimina la primera categoria de cada variable categorica
    # para evitar multicolinealidad perfecta (trampa de variables dummy).
    # Categoria de referencia por variable:
    #   occupation_status -> Employed
    #   product_type      -> Credit Card
    #   loan_intent       -> Business
    df_ohe = pd.get_dummies(df_clean, columns=CAT_COLS, drop_first=True, dtype=int)

    ohe_generadas = [c for c in df_ohe.columns
                     if any(c.startswith(cat + "_") for cat in CAT_COLS)]
    print(f"\n  One-Hot Encoding (drop_first=True):")
    print(f"    Columnas OHE generadas: {len(ohe_generadas)}")
    for col in ohe_generadas:
        print(f"      {col}")
    print(f"  Categorias de referencia: Employed | Credit Card | Business")
    print(f"  Shape post-OHE: {df_ohe.shape}")

    # 3. Separar features y target
    X = df_ohe.drop(columns=[TARGET])
    y = df_ohe[TARGET]
    feature_names = X.columns.tolist()

    # 4. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Split train/test (80/20, random_state=42):")
    print(f"    Train: {X_train.shape[0]:,} obs ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"    Test : {X_test.shape[0]:,} obs  ({X_test.shape[0]/len(X)*100:.0f}%)")

    # 5. StandardScaler — ajustado SOLO sobre train
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"\n  StandardScaler aplicado:")
    print(f"    fit() sobre X_train unicamente (evita data leakage del scaler)")
    print(f"    transform() aplicado a X_train y X_test")

    return X_train_sc, X_test_sc, y_train, y_test, feature_names, scaler, X_train, X_test


# =============================================================================
# MODELO A — REGRESION LINEAL (sklearn)
# =============================================================================

def entrenar_modelo(
    X_train_sc: np.ndarray,
    X_test_sc: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple:
    """
    Entrena LinearRegression sobre X_train escalado y calcula metricas
    sobre train Y test para diagnosticar overfitting/underfitting.

    Metricas calculadas:
    - MAE  (Mean Absolute Error): error promedio en las mismas unidades del
      target (USD). Interpretacion directa: en promedio el modelo se equivoca
      en $X USD.
    - RMSE (Root Mean Squared Error): penaliza errores grandes mas que el MAE
      por el cuadrado. Si RMSE >> MAE, hay predicciones muy malas en algunos
      casos.
    - R2   (coeficiente de determinacion): proporcion de la varianza de
      loan_amount explicada por el modelo. R2=0.80 significa que el 80% de
      la variabilidad del monto del prestamo esta explicada por las features.

    Diagnostico overfitting/underfitting:
    - Si R2_train >> R2_test: overfitting (memorizo el ruido)
    - Si R2_train ≈ R2_test y ambos bajos: underfitting (modelo insuficiente)
    - Si R2_train ≈ R2_test y ambos altos: buen ajuste (generaliza bien)

    Parameters
    ----------
    X_train_sc : np.ndarray — Features escaladas de entrenamiento
    X_test_sc  : np.ndarray — Features escaladas de test
    y_train    : pd.Series  — Target de entrenamiento
    y_test     : pd.Series  — Target de test

    Returns
    -------
    modelo   : LinearRegression ajustado
    y_pred   : np.ndarray predicciones sobre test
    metricas : dict con MAE, RMSE, R2 para train y test
    """
    separador = "=" * 60
    print(f"\n{separador}")
    print("MODELO A — REGRESION LINEAL (sklearn)")
    print(separador)

    modelo = LinearRegression()
    modelo.fit(X_train_sc, y_train)

    # Predicciones train y test
    y_pred_train = modelo.predict(X_train_sc)
    y_pred_test  = modelo.predict(X_test_sc)

    # Metricas train
    mae_train  = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train   = r2_score(y_train, y_pred_train)

    # Metricas test
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test   = r2_score(y_test, y_pred_test)

    metricas = {
        "MAE_train" : mae_train,  "MAE_test" : mae_test,
        "RMSE_train": rmse_train, "RMSE_test": rmse_test,
        "R2_train"  : r2_train,   "R2_test"  : r2_test,
    }

    print(f"\n  {'Metrica':<12} {'Train':>12} {'Test':>12} {'Diferencia':>12}")
    print(f"  {'-'*50}")
    print(f"  {'MAE':<12} ${mae_train:>10,.2f} ${mae_test:>10,.2f} "
          f"${abs(mae_test-mae_train):>10,.2f}")
    print(f"  {'RMSE':<12} ${rmse_train:>10,.2f} ${rmse_test:>10,.2f} "
          f"${abs(rmse_test-rmse_train):>10,.2f}")
    print(f"  {'R2':<12}  {r2_train:>10.4f}  {r2_test:>10.4f} "
          f" {abs(r2_test-r2_train):>10.4f}")

    # Diagnostico
    diff_r2 = r2_train - r2_test
    print(f"\n  DIAGNOSTICO:")
    if diff_r2 < 0.02:
        print(f"    R2_train - R2_test = {diff_r2:.4f} (< 0.02)")
        print(f"    Sin overfitting detectable. El modelo generaliza bien.")
    elif diff_r2 < 0.05:
        print(f"    R2_train - R2_test = {diff_r2:.4f} (0.02–0.05)")
        print(f"    Leve sobreajuste. Aceptable para un modelo lineal simple.")
    else:
        print(f"    R2_train - R2_test = {diff_r2:.4f} (> 0.05)")
        print(f"    Overfitting significativo. Considerar regularizacion.")

    if r2_test < 0.4:
        print(f"    R2_test = {r2_test:.4f} — posible underfitting.")
        print(f"    La relacion lineal puede ser insuficiente para este target.")

    return modelo, y_pred_test, metricas


# =============================================================================
# GUARDAR METRICAS
# =============================================================================

def guardar_metricas(metricas: dict) -> None:
    """
    Guarda MAE, RMSE y R2 de train y test en output/ej2_metricas_regresion.txt.

    Parameters
    ----------
    metricas : dict
        Diccionario con claves MAE_train, MAE_test, RMSE_train,
        RMSE_test, R2_train, R2_test.

    Returns
    -------
    None
    """
    ruta = OUTPUT_DIR / "ej2_metricas_regresion.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("Regresion Lineal — Metricas de Evaluacion\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Metrica':<12} {'Train':>14} {'Test':>14}\n")
        f.write("-" * 42 + "\n")
        f.write(f"{'MAE':<12} ${metricas['MAE_train']:>12,.2f} "
                f"${metricas['MAE_test']:>12,.2f}\n")
        f.write(f"{'RMSE':<12} ${metricas['RMSE_train']:>12,.2f} "
                f"${metricas['RMSE_test']:>12,.2f}\n")
        f.write(f"{'R2':<12}  {metricas['R2_train']:>12.4f} "
                f" {metricas['R2_test']:>12.4f}\n")
        f.write("\nInterpretacion:\n")
        f.write(f"  MAE_test = ${metricas['MAE_test']:,.2f} USD\n")
        f.write("  En promedio el modelo se equivoca en ese monto al predecir\n")
        f.write("  el monto del prestamo sobre datos no vistos.\n\n")
        f.write(f"  R2_test = {metricas['R2_test']:.4f}\n")
        f.write(f"  El modelo explica el {metricas['R2_test']*100:.1f}% de la\n")
        f.write("  variabilidad del monto del prestamo en el test set.\n\n")
        diff = metricas['R2_train'] - metricas['R2_test']
        f.write(f"  Diferencia R2 train-test = {diff:.4f}\n")
        if diff < 0.02:
            f.write("  Sin overfitting detectable.\n")
        elif diff < 0.05:
            f.write("  Leve sobreajuste — aceptable para modelo lineal.\n")
        else:
            f.write("  Overfitting significativo.\n")
    print(f"  Guardado: {ruta.name}")


# =============================================================================
# GRAFICO DE RESIDUOS
# =============================================================================

def graficar_residuos(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """
    Genera el grafico de residuos: valores predichos (eje X) vs residuos (eje Y).

    Un modelo OLS bien especificado debe mostrar residuos distribuidos
    aleatoriamente alrededor de 0 sin ningun patron sistematico.
    Patrones en los residuos indican:
    - Forma de abanico (heterocedasticidad): la varianza del error no es
      constante — supuesto OLS violado. Comun cuando hay outliers en features
      como savings_assets (detectado en Ejercicio 1).
    - Curvatura: la relacion no es lineal — el modelo necesita terminos
      polinomicos o transformaciones.
    - Puntos muy alejados: observaciones influyentes que afectan
      desproporcionadamente los coeficientes.

    Parameters
    ----------
    y_test : pd.Series  — Valores reales del test set
    y_pred : np.ndarray — Predicciones del modelo

    Returns
    -------
    None
        Guarda output/ej2_residuos.png
    """
    residuos = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Predichos vs Residuos
    axes[0].scatter(
        y_pred, residuos,
        alpha=0.3, s=8, color=sns.color_palette("muted")[0]
    )
    axes[0].axhline(0, color="crimson", linewidth=1.5, linestyle="--",
                    label="Residuo = 0")
    axes[0].set_xlabel("Valores Predichos (USD)", fontsize=10)
    axes[0].set_ylabel("Residuos (USD)", fontsize=10)
    axes[0].set_title("Residuos vs Valores Predichos", fontsize=11,
                      fontweight="bold")
    axes[0].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[0].legend(fontsize=9)

    # Anotacion del patron esperado
    axes[0].text(
        0.02, 0.96,
        "Distribucion aleatoria alrededor de 0\n= supuesto homocedasticidad cumplido",
        transform=axes[0].transAxes, fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7)
    )

    # Panel 2: Histograma de residuos
    axes[1].hist(
        residuos, bins=50,
        color=sns.color_palette("muted")[1],
        edgecolor="white", linewidth=0.5, alpha=0.85
    )
    axes[1].axvline(0, color="crimson", linewidth=1.5, linestyle="--",
                    label="Residuo = 0")
    axes[1].axvline(np.mean(residuos), color="navy", linewidth=1.2,
                    linestyle=":", label=f"Media: ${np.mean(residuos):,.0f}")
    axes[1].set_xlabel("Residuo (USD)", fontsize=10)
    axes[1].set_ylabel("Frecuencia", fontsize=10)
    axes[1].set_title("Distribucion de Residuos", fontsize=11, fontweight="bold")
    axes[1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    axes[1].legend(fontsize=9)

    fig.suptitle(
        "Analisis de Residuos — Regresion Lineal\nLoan Approval Data 2025",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej2_residuos.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")

    # Estadisticos de los residuos en consola
    print(f"\n  Estadisticos de residuos:")
    print(f"    Media   : ${np.mean(residuos):>10,.2f} USD  (ideal: 0)")
    print(f"    Std     : ${np.std(residuos):>10,.2f} USD")
    print(f"    Min/Max : ${residuos.min():>10,.2f} / ${residuos.max():,.2f} USD")


# =============================================================================
# GRAFICO DE COEFICIENTES
# =============================================================================

def graficar_coeficientes(
    modelo: LinearRegression,
    feature_names: list,
) -> None:
    """
    Genera un barplot horizontal de los 10 coeficientes con mayor valor
    absoluto del modelo estandarizado.

    Interpretacion de coeficientes estandarizados:
    Un coeficiente de +5,000 para 'annual_income' (estandarizado) significa:
    cuando annual_income aumenta 1 desviacion tipica, loan_amount aumenta
    $5,000 en promedio, manteniendo todo lo demas constante (ceteris paribus).

    Los coeficientes de variables OHE se interpretan respecto a la categoria
    de referencia:
    - occupation_status_Student: diferencia en loan_amount entre Student
      y Employed (categoria de referencia), ceteris paribus.

    Parameters
    ----------
    modelo        : LinearRegression ajustado sobre datos estandarizados
    feature_names : list de nombres de features post-OHE

    Returns
    -------
    None
        Guarda output/ej2_coeficientes.png
    """
    coefs = pd.Series(modelo.coef_, index=feature_names)
    top10 = coefs.abs().nlargest(10).index
    coefs_top = coefs[top10].sort_values()

    # Mapeo nombres OHE a nombres mas legibles
    nombres_legibles = {
        "occupation_status_Self-Employed": "Ocupacion: Self-Employed\n(vs Employed)",
        "occupation_status_Student"      : "Ocupacion: Student\n(vs Employed)",
        "product_type_Line of Credit"    : "Producto: Line of Credit\n(vs Credit Card)",
        "product_type_Personal Loan"     : "Producto: Personal Loan\n(vs Credit Card)",
        "loan_intent_Debt Consolidation" : "Intencion: Debt Consol.\n(vs Business)",
        "loan_intent_Education"          : "Intencion: Education\n(vs Business)",
        "loan_intent_Home Improvement"   : "Intencion: Home Improv.\n(vs Business)",
        "loan_intent_Medical"            : "Intencion: Medical\n(vs Business)",
        "loan_intent_Personal"           : "Intencion: Personal\n(vs Business)",
    }
    etiquetas = [nombres_legibles.get(n, n) for n in coefs_top.index]
    colores   = ["#2196F3" if v > 0 else "#E53935" for v in coefs_top.values]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(coefs_top)), coefs_top.values,
        color=colores, edgecolor="white", linewidth=0.5, alpha=0.85
    )
    ax.set_yticks(range(len(coefs_top)))
    ax.set_yticklabels(etiquetas, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente estandarizado (USD por 1 SD)", fontsize=10)
    ax.set_title(
        "Top-10 Coeficientes por Peso Absoluto\n"
        "Regresion Lineal Estandarizada — Loan Approval Data 2025",
        fontsize=11, fontweight="bold"
    )

    # Valores sobre las barras
    for bar, val in zip(bars, coefs_top.values):
        xpos = val + 50 if val >= 0 else val - 50
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", ha=ha, fontsize=8)

    # Leyenda de colores
    from matplotlib.patches import Patch
    leyenda = [
        Patch(facecolor="#2196F3", label="Efecto positivo sobre loan_amount"),
        Patch(facecolor="#E53935", label="Efecto negativo sobre loan_amount"),
    ]
    ax.legend(handles=leyenda, fontsize=8, loc="lower right")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej2_coeficientes.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")


# =============================================================================
# SUMMARY OLS — statsmodels (inferencia estadistica)
# =============================================================================

def summary_ols(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list,
) -> None:
    """
    Ajusta OLS con statsmodels sobre datos NO estandarizados para obtener
    inferencia estadistica: p-values, t-statistics, intervalos de confianza.

    Por que NO estandarizado aqui:
    Los coeficientes de statsmodels se interpretan en las unidades originales
    de cada variable — igual que la tabla 'reg' en STATA. Si estandarizaras,
    los coeficientes perderian su interpretacion economica directa.

    Interpretacion de la tabla (como en STATA):
    - coef    : beta estimado. Cambio en loan_amount por una unidad de la feature,
                ceteris paribus.
    - std err : error estandar del estimador. Incertidumbre en la estimacion.
    - t       : coef / std_err. Si |t| > 1.96, significativo al 5%.
    - P>|t|   : p-value. Prob. de observar ese beta si H0: beta=0 fuera verdad.
                p < 0.05 -> rechaza H0 -> variable estadisticamente significativa.
    - [0.025, 0.975]: intervalo de confianza al 95% del coeficiente.

    Parameters
    ----------
    X_train      : pd.DataFrame — Features originales (no estandarizadas) de train
    y_train      : pd.Series    — Target de train
    feature_names: list         — Nombres de features post-OHE

    Returns
    -------
    None
        Guarda output/ej2_summary_ols.txt
    """
    separador = "=" * 60
    print(f"\n{separador}")
    print("SUMMARY OLS — statsmodels (inferencia estadistica)")
    print(separador)

    # Agregar constante (intercepto) — equivalente a _cons en STATA
    X_sm = sm.add_constant(X_train)
    modelo_sm = sm.OLS(y_train, X_sm).fit()

    # Mostrar en consola
    print(modelo_sm.summary())

    # Variables significativas al 5%
    pvals = modelo_sm.pvalues.drop("const")
    sig   = pvals[pvals < 0.05].sort_values()
    no_sig = pvals[pvals >= 0.05].sort_values()

    print(f"\n  Variables SIGNIFICATIVAS al 5% (p < 0.05): {len(sig)}/{len(pvals)}")
    for var, p in sig.items():
        beta = modelo_sm.params[var]
        print(f"    {var:<40} beta={beta:>10,.2f}  p={p:.4f}")

    if len(no_sig) > 0:
        print(f"\n  Variables NO significativas (p >= 0.05): {len(no_sig)}")
        for var, p in no_sig.items():
            beta = modelo_sm.params[var]
            print(f"    {var:<40} beta={beta:>10,.2f}  p={p:.4f}")

    # R2 ajustado
    print(f"\n  R2          : {modelo_sm.rsquared:.4f}")
    print(f"  R2 ajustado : {modelo_sm.rsquared_adj:.4f}")
    print(f"  Diferencia  : {modelo_sm.rsquared - modelo_sm.rsquared_adj:.4f}")
    print(f"  (Diferencia pequeña indica que las features aportaron valor real)")

    # Guardar summary completo
    ruta = OUTPUT_DIR / "ej2_summary_ols.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("Summary OLS — statsmodels\n")
        f.write("Equivalente a 'reg' en STATA\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(modelo_sm.summary()))
        f.write("\n\n")
        f.write("GUIA DE INTERPRETACION\n")
        f.write("=" * 60 + "\n")
        f.write("coef    : beta estimado. Cambio en loan_amount (USD) por una\n")
        f.write("          unidad de la feature, ceteris paribus.\n")
        f.write("std err : error estandar. Incertidumbre en la estimacion.\n")
        f.write("t       : coef / std_err. |t| > 1.96 -> significativo al 5%.\n")
        f.write("P>|t|   : p-value. < 0.05 -> rechaza H0: beta=0.\n")
        f.write("[0.025] : limite inferior IC 95% del coeficiente.\n")
        f.write("[0.975] : limite superior IC 95% del coeficiente.\n")
        f.write("\nVariables NO significativas (p >= 0.05):\n")
        for var, p in no_sig.items():
            f.write(f"  {var:<40} p={p:.4f}\n")
        if len(no_sig) == 0:
            f.write("  Todas las variables son significativas al 5%.\n")
    print(f"\n  Guardado: {ruta.name}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EJERCICIO 2 — REGRESION LINEAL CON SCIKIT-LEARN")
    print("Dataset: Loan Approval Data 2025")
    print("=" * 60)

    # Carga
    print(f"\nCargando dataset desde: {DATA_PATH.relative_to(BASE_DIR)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Preprocesamiento
    X_train_sc, X_test_sc, y_train, y_test, feature_names, scaler, X_train, X_test = \
        preprocesar(df)

    # Modelo sklearn
    modelo, y_pred, metricas = entrenar_modelo(
        X_train_sc, X_test_sc, y_train, y_test
    )

    # Salidas
    print("\n" + "=" * 60)
    print("GENERANDO SALIDAS")
    print("=" * 60)
    guardar_metricas(metricas)
    graficar_residuos(y_test, y_pred)
    graficar_coeficientes(modelo, feature_names)
    summary_ols(X_train, y_train, feature_names)

    # Checklist final
    print("\n" + "=" * 60)
    print("SALIDAS GENERADAS EN output/")
    print("=" * 60)
    salidas = [
        "ej2_metricas_regresion.txt",
        "ej2_residuos.png",
        "ej2_coeficientes.png",
        "ej2_summary_ols.txt",
    ]
    for s in salidas:
        existe = (OUTPUT_DIR / s).exists()
        estado = "OK" if existe else "FALTA"
        print(f"  [{estado}] {s}")

    print("\nEjercicio 2 completado.")
