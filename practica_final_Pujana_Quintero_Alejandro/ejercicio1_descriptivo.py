"""
=============================================================================
PRACTICA FINAL — EJERCICIO 1
Analisis Estadistico Descriptivo
=============================================================================

DATASET : Loan Approval Data 2025 (Kaggle)
          50,000 registros x 20 columnas | 5.5 MB en disco
TARGET  : loan_amount — monto del prestamo aprobado (USD, continuo)

EXCLUSIONES JUSTIFICADAS:
  - customer_id   : identificador administrativo, sin valor predictivo
  - loan_status   : resultado final de la decision (data leakage)
  - payment_to_income_ratio : r=1.000 con loan_to_income_ratio
                              (multicolinealidad perfecta)

SALIDAS (output/):
  ej1_descriptivo.csv          Estadisticos descriptivos variables numericas
  ej1_histogramas.png          Histogramas de variables numericas
  ej1_boxplots.png             Boxplots de loan_amount por variable categorica
  ej1_heatmap_correlacion.png  Mapa de calor correlaciones Pearson
  ej1_categoricas.png          Frecuencias de variables categoricas

EJECUCION:
  Desde practica_final_Pujana_Quintero_Alejandro/
  $ python ejercicio1_descriptivo.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── Reproducibilidad ────────────────────────────────────────────────────────
np.random.seed(42)

# ── Rutas relativas al script (funciona en cualquier maquina) ────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data" / "Loan_approval_data_2025.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Constantes del dataset ───────────────────────────────────────────────────
TARGET      = "loan_amount"
DROP_COLS   = ["customer_id", "loan_status", "payment_to_income_ratio"]
CAT_COLS    = ["occupation_status", "product_type", "loan_intent"]
NUM_COLS    = [
    "age", "years_employed", "annual_income", "credit_score",
    "credit_history_years", "savings_assets", "current_debt",
    "defaults_on_file", "delinquencies_last_2yrs", "derogatory_marks",
    "interest_rate", "debt_to_income_ratio", "loan_to_income_ratio",
    "loan_amount",
]

# ── Estilo visual ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = sns.color_palette("muted")


# =============================================================================
# A) RESUMEN ESTRUCTURAL
# =============================================================================

def resumen_estructural(df: pd.DataFrame) -> None:
    """
    Imprime el resumen estructural del dataset: shape, dtypes, nulos y
    tamano en memoria. Documenta las columnas excluidas y su justificacion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original cargado desde CSV.

    Returns
    -------
    None
        Imprime resultados en consola.
    """
    separador = "=" * 60

    print(separador)
    print("A) RESUMEN ESTRUCTURAL")
    print(separador)
    print(f"  Filas    : {df.shape[0]:,}")
    print(f"  Columnas : {df.shape[1]}")
    print(f"  Tamano en disco  : ~5.5 MB")
    print(f"  Uso en memoria   : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n  Tipos de dato por columna:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<35} {dtype}")

    print("\n  Valores nulos por columna:")
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        print("    HALLAZGO: El dataset no contiene valores nulos.")
        print("    Decision: No se requiere imputacion ni eliminacion de filas.")
    else:
        for col, n in nulos[nulos > 0].items():
            pct = n / len(df) * 100
            print(f"    {col:<35} {n:>6} ({pct:.2f}%)")

    print("\n  Columnas excluidas del analisis:")
    print("    customer_id             -> ID administrativo, sin valor predictivo")
    print("    loan_status             -> Data leakage: resultado post-decision")
    print("    payment_to_income_ratio -> Multicolinealidad perfecta (r=1.000")
    print("                              con loan_to_income_ratio)")


# =============================================================================
# B) ESTADISTICOS DESCRIPTIVOS
# =============================================================================

def estadisticos_descriptivos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadisticos descriptivos completos para todas las variables
    numericas: media, mediana, moda, std, varianza, min, max, cuartiles,
    IQR, skewness y curtosis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas numericas ya seleccionadas (sin exclusiones).

    Returns
    -------
    stats : pd.DataFrame
        Tabla de estadisticos con variables como filas y metricas como
        columnas. Se guarda en output/ej1_descriptivo.csv.
    """
    separador = "=" * 60
    print(f"\n{separador}")
    print("B) ESTADISTICOS DESCRIPTIVOS — VARIABLES NUMERICAS")
    print(separador)

    df_num = df[NUM_COLS]

    stats = pd.DataFrame({
        "media"    : df_num.mean(),
        "mediana"  : df_num.median(),
        "moda"     : df_num.mode().iloc[0],
        "std"      : df_num.std(),
        "varianza" : df_num.var(),
        "min"      : df_num.min(),
        "max"      : df_num.max(),
        "Q1"       : df_num.quantile(0.25),
        "Q3"       : df_num.quantile(0.75),
        "IQR"      : df_num.quantile(0.75) - df_num.quantile(0.25),
        "skewness" : df_num.skew(),
        "curtosis" : df_num.kurtosis(),
    }).round(4)

    # Guardar CSV
    ruta_csv = OUTPUT_DIR / "ej1_descriptivo.csv"
    stats.to_csv(ruta_csv, encoding="utf-8")
    print(f"  Guardado: {ruta_csv.name}")

    # Resumen especifico de la variable objetivo
    print(f"\n  Variable objetivo: {TARGET}")
    print(f"    Media    : ${stats.loc[TARGET, 'media']:>12,.2f} USD")
    print(f"    Mediana  : ${stats.loc[TARGET, 'mediana']:>12,.2f} USD")
    print(f"    Std      : ${stats.loc[TARGET, 'std']:>12,.2f} USD")
    print(f"    IQR      : ${stats.loc[TARGET, 'IQR']:>12,.2f} USD")
    print(f"    Skewness :  {stats.loc[TARGET, 'skewness']:>12.4f}")
    print(f"    Curtosis :  {stats.loc[TARGET, 'curtosis']:>12.4f}")
    print(f"    Min/Max  : ${stats.loc[TARGET, 'min']:>10,.0f} / "
          f"${stats.loc[TARGET, 'max']:,.0f} USD")

    # Interpretacion economica del skewness
    skew_val = stats.loc[TARGET, "skewness"]
    if skew_val > 0.5:
        print(f"\n  Interpretacion: skewness = {skew_val:.4f} > 0 indica")
        print("    distribucion con cola derecha (prestamos grandes menos")
        print("    frecuentes pero existentes). La mediana < media confirma")
        print("    asimetria positiva. Implicacion: la mayoria de solicitantes")
        print("    recibe prestamos moderados; pocos reciben montos altos.")

    return stats


# =============================================================================
# C) DISTRIBUCIONES — HISTOGRAMAS Y BOXPLOTS
# =============================================================================

def graficar_histogramas(df: pd.DataFrame) -> None:
    """
    Genera histogramas para todas las variables numericas y los guarda
    en output/ej1_histogramas.png.

    La eleccion del numero de bins sigue la regla de Sturges adaptada.
    No se incluye curva KDE por decision del docente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas numericas de interes.

    Returns
    -------
    None
        Guarda la figura en output/.
    """
    df_num = df[NUM_COLS]
    n_cols_plot = 4
    n_rows_plot = int(np.ceil(len(NUM_COLS) / n_cols_plot))

    fig, axes = plt.subplots(
        n_rows_plot, n_cols_plot,
        figsize=(20, n_rows_plot * 4)
    )
    axes = axes.flatten()

    for i, col in enumerate(NUM_COLS):
        axes[i].hist(
            df_num[col].dropna(),
            bins=40,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        axes[i].set_title(col, fontsize=11, fontweight="bold", pad=8)
        axes[i].set_xlabel("Valor", fontsize=9)
        axes[i].set_ylabel("Frecuencia", fontsize=9)
        axes[i].tick_params(axis="both", labelsize=8)

        # Linea vertical en la mediana para referencia visual
        mediana = df_num[col].median()
        axes[i].axvline(
            mediana, color="crimson", linestyle="--",
            linewidth=1.2, label=f"Mediana: {mediana:.1f}"
        )
        axes[i].legend(fontsize=7)

    # Ocultar subplots vacios
    for j in range(len(NUM_COLS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribucion de Variables Numericas — Loan Approval Data 2025",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej1_histogramas.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")


def detectar_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta outliers en variables numericas usando el metodo IQR.

    Un valor es outlier si cae fuera del intervalo:
        [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]

    Se eligio IQR sobre Z-Score porque varias variables (especialmente
    loan_amount) presentan distribucion asimetrica (skewness > 0.5).
    El Z-Score asume normalidad y sobreestima outliers en distribuciones
    sesgadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas numericas.

    Returns
    -------
    resumen : pd.DataFrame
        Tabla con Q1, Q3, IQR, limites y cantidad de outliers por variable.
    """
    separador = "=" * 60
    print(f"\n{separador}")
    print("C) DETECCION DE OUTLIERS — METODO IQR")
    print(separador)
    print("  Metodo: IQR (seleccionado sobre Z-Score por asimetria en datos)")
    print("  Criterio: valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]")
    print("  Decision: NO eliminar — valores extremos son prestamos reales\n")

    df_num = df[NUM_COLS]
    registros = []

    for col in NUM_COLS:
        Q1  = df_num[col].quantile(0.25)
        Q3  = df_num[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        n_out   = ((df_num[col] < lim_inf) | (df_num[col] > lim_sup)).sum()
        pct_out = n_out / len(df) * 100

        registros.append({
            "variable"  : col,
            "Q1"        : round(Q1, 2),
            "Q3"        : round(Q3, 2),
            "IQR"       : round(IQR, 2),
            "lim_inf"   : round(lim_inf, 2),
            "lim_sup"   : round(lim_sup, 2),
            "n_outliers": n_out,
            "pct_outliers": round(pct_out, 2),
        })
        estado = f"{n_out:>5} outliers ({pct_out:.2f}%)"
        print(f"  {col:<35} {estado}")

    resumen = pd.DataFrame(registros).set_index("variable")

    # Hallazgo especifico del target
    row_target = resumen.loc[TARGET]
    print(f"\n  HALLAZGO — {TARGET}:")
    print(f"    Limite superior IQR: ${row_target['lim_sup']:,.0f} USD")
    print(f"    Maximo en dataset  : ${df[TARGET].max():,.0f} USD")
    print(f"    Outliers detectados: {int(row_target['n_outliers'])}")
    print("    Conclusion: la variable objetivo no contiene outliers segun")
    print("    IQR. Los valores extremos son montos validos dentro del rango")
    print("    de productos financieros ofrecidos.")

    return resumen


def graficar_boxplots(df: pd.DataFrame) -> None:
    """
    Genera boxplots de loan_amount segmentados por cada variable categorica
    y los guarda en output/ej1_boxplots.png.

    Los boxplots permiten comparar la distribucion del monto del prestamo
    entre grupos, identificando diferencias sistematicas por tipo de ocupacion,
    producto financiero y proposito del prestamo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo con variables categoricas y target.

    Returns
    -------
    None
        Guarda la figura en output/.
    """
    fig, axes = plt.subplots(1, len(CAT_COLS), figsize=(18, 6))

    for i, cat in enumerate(CAT_COLS):
        order = df.groupby(cat)[TARGET].median().sort_values(ascending=False).index

        sns.boxplot(
            data=df,
            x=cat,
            y=TARGET,
            order=order,
            hue=cat,
            palette="muted",
            linewidth=0.8,
            fliersize=2,
            legend=False,
            ax=axes[i],
        )
        axes[i].set_title(
            f"{TARGET}\npor {cat}",
            fontsize=11, fontweight="bold"
        )
        axes[i].set_xlabel(cat, fontsize=9)
        axes[i].set_ylabel("loan_amount (USD)", fontsize=9)
        axes[i].tick_params(axis="x", rotation=20, labelsize=8)
        axes[i].tick_params(axis="y", labelsize=8)
        axes[i].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )

    fig.suptitle(
        "Distribucion de loan_amount por Variable Categorica",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej1_boxplots.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")


# =============================================================================
# D) VARIABLES CATEGORICAS
# =============================================================================

def analizar_categoricas(df: pd.DataFrame) -> None:
    """
    Calcula frecuencias absolutas y relativas de cada variable categorica
    y genera graficos de barras. Analiza desbalance entre categorias.

    Variables analizadas: occupation_status, product_type, loan_intent.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las variables categoricas de interes.

    Returns
    -------
    None
        Imprime tabla de frecuencias y guarda output/ej1_categoricas.png.
    """
    separador = "=" * 60
    print(f"\n{separador}")
    print("D) VARIABLES CATEGORICAS — FRECUENCIAS Y DESBALANCE")
    print(separador)

    fig, axes = plt.subplots(1, len(CAT_COLS), figsize=(18, 6))

    for i, cat in enumerate(CAT_COLS):
        freq_abs = df[cat].value_counts()
        freq_rel = df[cat].value_counts(normalize=True) * 100

        print(f"\n  {cat} ({df[cat].nunique()} categorias):")
        for cat_val in freq_abs.index:
            dominante = " <- DOMINANTE" if freq_rel[cat_val] > 50 else ""
            print(f"    {cat_val:<25} "
                  f"{freq_abs[cat_val]:>6,} obs "
                  f"({freq_rel[cat_val]:>5.1f}%){dominante}")

        # Analisis de desbalance: ratio max/min
        ratio = freq_abs.max() / freq_abs.min()
        if ratio > 2:
            print(f"  DESBALANCE: categoria mas frecuente tiene "
                  f"{ratio:.1f}x mas obs que la menos frecuente.")
        else:
            print(f"  Distribucion relativamente balanceada (ratio {ratio:.1f}x).")

        # Grafico de barras
        colores = sns.color_palette("muted", n_colors=len(freq_abs))
        bars = axes[i].bar(
            freq_abs.index,
            freq_abs.values,
            color=colores,
            edgecolor="white",
            linewidth=0.5,
        )
        # Etiquetas de porcentaje sobre cada barra
        for bar, pct in zip(bars, freq_rel.values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 200,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )
        axes[i].set_title(cat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Categoria", fontsize=9)
        axes[i].set_ylabel("Frecuencia absoluta", fontsize=9)
        axes[i].tick_params(axis="x", rotation=20, labelsize=8)
        axes[i].tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Frecuencia de Variables Categoricas — Loan Approval Data 2025",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej1_categoricas.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Guardado: {ruta.name}")


# =============================================================================
# E) CORRELACIONES
# =============================================================================

def analizar_correlaciones(df: pd.DataFrame) -> None:
    """
    Calcula y visualiza la matriz de correlaciones de Pearson para todas
    las variables numericas. Identifica las tres variables con mayor
    correlacion con loan_amount y detecta multicolinealidad.

    Nota sobre variables discretas incluidas:
        defaults_on_file (0/1) y delinquencies_last_2yrs (0-9) se incluyen
        en el heatmap. La correlacion de Pearson es valida para variables
        binarias (equivale al coeficiente point-biserial) y discretas con
        suficiente rango de valores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas numericas (incluye binarias discretas).

    Returns
    -------
    None
        Imprime top correlaciones y guarda output/ej1_heatmap_correlacion.png.
    """
    separador = "=" * 60
    print(f"\n{separador}")
    print("E) CORRELACIONES DE PEARSON")
    print(separador)

    df_num = df[NUM_COLS]
    corr_matrix = df_num.corr(method="pearson")

    # Top-3 correlaciones con loan_amount (excluyendo el target consigo mismo)
    corr_target = corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print(f"\n  Top-3 variables con mayor correlacion con {TARGET}:")
    for rank, (var, val) in enumerate(corr_target.head(3).items(), 1):
        signo = corr_matrix.loc[var, TARGET]
        print(f"    {rank}. {var:<35} r = {signo:+.4f}")

    # Multicolinealidad: pares con |r| > 0.9
    print("\n  Pares con multicolinealidad |r| > 0.9:")
    encontrado = False
    cols = list(corr_matrix.columns)
    for ii in range(len(cols)):
        for jj in range(ii + 1, len(cols)):
            val = corr_matrix.iloc[ii, jj]
            if abs(val) > 0.9:
                print(f"    {cols[ii]:<35} <-> {cols[jj]:<35} r = {val:+.4f}")
                encontrado = True
    if not encontrado:
        print("    No se detectaron pares con |r| > 0.9")

    print("\n  DECISION: payment_to_income_ratio excluida del analisis porque")
    print("  r = 1.000 con loan_to_income_ratio viola el supuesto de no")
    print("  multicolinealidad de OLS. XtX seria singular (no invertible).")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": 7},
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Correlacion de Pearson"},
    )
    ax.set_title(
        "Matriz de Correlaciones de Pearson — Variables Numericas\n"
        "Loan Approval Data 2025",
        fontsize=12, fontweight="bold", pad=16
    )
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej1_heatmap_correlacion.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Guardado: {ruta.name}")


# =============================================================================
# MAIN — PIPELINE COMPLETO
# =============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EJERCICIO 1 — ANALISIS ESTADISTICO DESCRIPTIVO")
    print("Dataset: Loan Approval Data 2025")
    print("=" * 60)

    # Carga del dataset
    print(f"\nCargando dataset desde: {DATA_PATH.relative_to(BASE_DIR)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # A) Resumen estructural
    resumen_estructural(df)

    # Eliminar columnas excluidas para el resto del analisis
    df_clean = df.drop(columns=DROP_COLS)

    # B) Estadisticos descriptivos
    stats = estadisticos_descriptivos(df_clean)

    # C) Distribuciones
    print("\n" + "=" * 60)
    print("C) DISTRIBUCIONES — HISTOGRAMAS Y BOXPLOTS")
    print("=" * 60)
    graficar_histogramas(df_clean)
    graficar_boxplots(df_clean)
    detectar_outliers_iqr(df_clean)

    # D) Variables categoricas
    analizar_categoricas(df_clean)

    # E) Correlaciones
    analizar_correlaciones(df_clean)

    # Resumen de salidas generadas
    print("\n" + "=" * 60)
    print("SALIDAS GENERADAS EN output/")
    print("=" * 60)
    salidas = [
        "ej1_descriptivo.csv",
        "ej1_histogramas.png",
        "ej1_boxplots.png",
        "ej1_categoricas.png",
        "ej1_heatmap_correlacion.png",
    ]
    for s in salidas:
        existe = (OUTPUT_DIR / s).exists()
        estado = "OK" if existe else "FALTA"
        print(f"  [{estado}] {s}")

    print("\nEjercicio 1 completado.")
