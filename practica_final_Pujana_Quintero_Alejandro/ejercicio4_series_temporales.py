"""
=============================================================================
PRACTICA FINAL — EJERCICIO 4
Analisis y Descomposicion de Series Temporales
=============================================================================

DESCRIPCION
-----------
En este ejercicio trabajamos con una serie temporal sintetica generada con
semilla fija. Las tareas son:

  1. Visualizar la serie completa.
  2. Descomponerla en Tendencia, Estacionalidad y Residuo.
  3. Analizar el residuo para evaluar si se comporta como ruido ideal.

LIBRERIAS UTILIZADAS
--------------------
  - numpy, pandas
  - matplotlib, seaborn
  - statsmodels (seasonal_decompose, adfuller, plot_acf, plot_pacf)
  - scipy.stats (jarque_bera, norm)

  Nota sobre tests de normalidad:
  Se usa Jarque-Bera en lugar de Shapiro-Wilk porque este ultimo esta
  limitado a n < 5,000 observaciones. La serie tiene 2,191 puntos pero
  el residuo post-descomposicion pierde ~365 en cada extremo, dejando
  ~1,461 valores validos. Jarque-Bera no tiene limite de muestra y es
  el test estandar en series temporales economicas.

COMPONENTES CONOCIDOS DE LA SERIE (generados con semilla=42):
  - Tendencia : lineal, pendiente=0.05/dia, intercepto=50
  - Estacional: amplitud ~15 unidades, periodo=365.25 dias
  - Ciclo     : amplitud=8 unidades, periodo~4 anos (1461 dias)
  - Ruido     : gaussiano, media=0, sigma=3.5

SALIDAS (output/):
  ej4_serie_original.png    Grafico de la serie completa
  ej4_descomposicion.png    4 subgraficos de descomposicion
  ej4_acf_pacf.png          ACF y PACF del residuo
  ej4_histograma_ruido.png  Histograma del residuo + curva normal teorica
  ej4_analisis.txt          Estadisticos numericos y resultados de tests

EJECUCION:
  Desde practica_final_Pujana_Quintero_Alejandro/
  $ python ejercicio4_series_temporales.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import jarque_bera, norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# ── Rutas ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Estilo visual ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLOR_SERIE    = "#1976D2"
COLOR_RESIDUO  = "#E53935"
COLOR_NORMAL   = "#43A047"


# =============================================================================
# GENERACION DE LA SERIE TEMPORAL SINTETICA — NO MODIFICAR ESTE BLOQUE
# =============================================================================

def generar_serie_temporal(semilla=42):
    """
    Genera una serie temporal sintetica con componentes conocidos.

    La serie tiene:
      - Una tendencia lineal creciente.
      - Estacionalidad anual (periodo 365 dias).
      - Ciclos de largo plazo (periodo ~4 anos).
      - Ruido gaussiano.

    Parameters
    ----------
    semilla : int
        Semilla aleatoria para reproducibilidad (NO modificar).

    Returns
    -------
    serie : pd.Series
        Serie con indice DatetimeIndex diario (2018-01-01 a 2023-12-31).
    """
    rng = np.random.default_rng(semilla)

    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    tendencia      = 0.05 * t + 50
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) \
                   +  6 * np.cos(4 * np.pi * t / 365.25)
    ciclo          = 8 * np.sin(2 * np.pi * t / 1461)
    ruido          = rng.normal(loc=0, scale=3.5, size=n)

    valores = tendencia + estacionalidad + ciclo + ruido
    serie   = pd.Series(valores, index=fechas, name="valor")
    return serie


# =============================================================================
# TAREA 1 — Visualizar la serie completa
# =============================================================================

def visualizar_serie(serie):
    """
    Genera y guarda un grafico de la serie temporal completa con anotaciones
    de los componentes teoricos conocidos.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal con indice DatetimeIndex.

    Returns
    -------
    None
        Guarda output/ej4_serie_original.png
    """
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(serie.index, serie.values,
            color=COLOR_SERIE, linewidth=0.8, alpha=0.9, label="Serie observada")

    # Media movil de 365 dias para visualizar la tendencia subyacente
    ma_365 = serie.rolling(window=365, center=True).mean()
    ax.plot(ma_365.index, ma_365.values,
            color="crimson", linewidth=1.8, linestyle="--",
            alpha=0.85, label="Media movil 365 dias (tendencia)")

    ax.set_title(
        "Serie Temporal Sintetica (semilla=42) \n"
        "Componentes: Tendencia lineal + Estacionalidad anual + Ciclo ~4 anos + Ruido",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Fecha", fontsize=10)
    ax.set_ylabel("Valor", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.4)

    # Anotaciones de los anos para referencia
    for anio in range(2018, 2024):
        ax.axvline(pd.Timestamp(f"{anio}-01-01"),
                   color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej4_serie_original.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")


# =============================================================================
# TAREA 2 — Descomposicion de la serie
# =============================================================================

def descomponer_serie(serie):
    """
    Descompone la serie en Tendencia, Estacionalidad y Residuo usando
    seasonal_decompose con modelo aditivo y periodo=365.

    Modelo aditivo: Y(t) = Tendencia(t) + Estacionalidad(t) + Residuo(t)
    Se usa modelo aditivo porque los componentes se suman (no se multiplican)
    y la amplitud de la estacionalidad no crece con el nivel de la serie.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal con indice DatetimeIndex.

    Returns
    -------
    resultado : DecomposeResult
        Objeto con atributos .trend, .seasonal, .resid y .observed.
    """
    resultado = seasonal_decompose(serie, model="additive", period=365)

    fig = resultado.plot()
    fig.set_size_inches(14, 10)

    # Mejorar titulos y estilos de cada subplot
    titulos = ["Serie original", "Tendencia", "Estacionalidad", "Residuo"]
    for i, ax in enumerate(fig.axes):
        ax.set_title(titulos[i], fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i < len(fig.axes) - 1:
            ax.set_xlabel("")

    fig.suptitle(
        "Descomposicion Aditiva de la Serie Temporal\n"
        "seasonal_decompose(model='additive', period=365)",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    ruta = OUTPUT_DIR / "ej4_descomposicion.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")

    # Estadisticos de cada componente en consola
    tend_limpia = resultado.trend.dropna()
    print(f"\n  Componentes descompuestos:")
    print(f"    Tendencia  — min: {tend_limpia.min():.2f} | "
          f"max: {tend_limpia.max():.2f} | "
          f"rango: {tend_limpia.max() - tend_limpia.min():.2f}")
    print(f"    Estacional — amplitud: {resultado.seasonal.max():.2f} | "
          f"periodo: 365 dias")
    print(f"    Residuo    — media: {resultado.resid.mean():.4f} | "
          f"std: {resultado.resid.std():.4f}")

    return resultado


# =============================================================================
# TAREA 3 — Analisis del residuo (ruido)
# =============================================================================

def analizar_residuo(residuo):
    """
    Analiza el componente residuo de la descomposicion para determinar si
    se comporta como ruido blanco gaussiano ideal.

    Un ruido ideal cumple:
      - Media = 0
      - Varianza constante (homocedasticidad)
      - Sin autocorrelacion (independencia)
      - Distribucion normal

    Tests aplicados:
      - Jarque-Bera: normalidad. H0: residuos normales. p > 0.05 no rechaza H0.
        Elegido sobre Shapiro-Wilk porque este tiene limite n < 5000.
      - ADF (Augmented Dickey-Fuller): estacionariedad. H0: raiz unitaria (no
        estacionario). p < 0.05 rechaza H0 -> residuo es estacionario.
      - ACF/PACF: detectan autocorrelacion. Barras dentro del IC de confianza
        (linea azul discontinua) indican ausencia de autocorrelacion significativa.

    Parameters
    ----------
    residuo : pd.Series
        Componente residuo de seasonal_decompose (puede contener NaN en extremos).

    Returns
    -------
    None
        Guarda: ej4_acf_pacf.png, ej4_histograma_ruido.png, ej4_analisis.txt
    """
    # Limpiar NaN generados por seasonal_decompose en los extremos
    residuo_limpio = residuo.dropna()

    # ── Estadisticos basicos ─────────────────────────────────────────────────
    media     = residuo_limpio.mean()
    std       = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis  = residuo_limpio.kurtosis()
    n_obs     = len(residuo_limpio)

    print(f"\n  Estadisticos del residuo (n={n_obs:,}):")
    print(f"    Media     : {media:.6f}  (ideal: 0)")
    print(f"    Std       : {std:.4f}   (sigma teorico: 3.5)")
    print(f"    Asimetria : {asimetria:.4f}  (ideal: 0)")
    print(f"    Curtosis  : {curtosis:.4f}  (ideal: 0 en exceso)")

    # ── Test Jarque-Bera ─────────────────────────────────────────────────────
    jb_stat, jb_p = jarque_bera(residuo_limpio)
    print(f"\n  Test Jarque-Bera (normalidad):")
    print(f"    Estadistico : {jb_stat:.4f}")
    print(f"    p-value     : {jb_p:.6f}")
    if jb_p > 0.05:
        print(f"    Conclusion  : No se rechaza H0 (p={jb_p:.4f} > 0.05)")
        print(f"                  El residuo sigue una distribucion normal.")
    else:
        print(f"    Conclusion  : Se rechaza H0 (p={jb_p:.4f} < 0.05)")
        print(f"                  El residuo NO sigue una distribucion normal estricta.")

    # ── Test ADF (estacionariedad) ────────────────────────────────────────────
    resultado_adf = adfuller(residuo_limpio, autolag="AIC")
    adf_stat = resultado_adf[0]
    adf_p    = resultado_adf[1]
    print(f"\n  Test ADF (estacionariedad):")
    print(f"    Estadistico ADF : {adf_stat:.4f}")
    print(f"    p-value         : {adf_p:.6f}")
    if adf_p < 0.05:
        print(f"    Conclusion : Rechaza H0 (p={adf_p:.4f} < 0.05)")
        print(f"                 El residuo ES estacionario (no hay raiz unitaria).")
    else:
        print(f"    Conclusion : No rechaza H0 (p={adf_p:.4f} >= 0.05)")
        print(f"                 El residuo NO es estacionario.")

    # ── Grafico ACF y PACF ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(
        residuo_limpio, lags=40, alpha=0.05, ax=axes[0],
        color=COLOR_RESIDUO, vlines_kwargs={"colors": COLOR_RESIDUO}
    )
    axes[0].set_title(
        "ACF del Residuo\n(barras dentro del IC = sin autocorrelacion)",
        fontsize=10, fontweight="bold"
    )
    axes[0].set_xlabel("Lag (dias)", fontsize=9)
    axes[0].set_ylabel("Autocorrelacion", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    plot_pacf(
        residuo_limpio, lags=40, alpha=0.05, ax=axes[1],
        color=COLOR_SERIE, vlines_kwargs={"colors": COLOR_SERIE},
        method="ywm"
    )
    axes[1].set_title(
        "PACF del Residuo\n(barras dentro del IC = sin autocorrelacion parcial)",
        fontsize=10, fontweight="bold"
    )
    axes[1].set_xlabel("Lag (dias)", fontsize=9)
    axes[1].set_ylabel("Autocorrelacion parcial", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "Autocorrelacion y Autocorrelacion Parcial del Residuo",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    ruta_acf = OUTPUT_DIR / "ej4_acf_pacf.png"
    fig.savefig(ruta_acf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Guardado: {ruta_acf.name}")

    # ── Histograma + curva normal teorica ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        residuo_limpio, bins=50,
        color=COLOR_RESIDUO, edgecolor="white",
        linewidth=0.5, alpha=0.75,
        density=True,          # Normaliza para comparar con PDF
        label="Residuo observado"
    )

    # Curva normal teorica con media y std del residuo
    x_rango = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 300)
    pdf_normal = norm.pdf(x_rango, loc=media, scale=std)
    ax.plot(x_rango, pdf_normal,
            color=COLOR_NORMAL, linewidth=2.5,
            label=f"Normal teorica N({media:.2f}, {std:.2f})")

    # Curva normal con sigma teorico conocido (3.5)
    pdf_teorica = norm.pdf(x_rango, loc=0, scale=3.5)
    ax.plot(x_rango, pdf_teorica,
            color="navy", linewidth=1.5, linestyle="--",
            label="Normal ideal N(0, 3.5) — sigma teorico")

    ax.axvline(media, color=COLOR_RESIDUO, linewidth=1.2,
               linestyle=":", label=f"Media residuo: {media:.4f}")
    ax.axvline(0, color="black", linewidth=0.8,
               linestyle="-", alpha=0.5, label="Media ideal = 0")

    ax.set_xlabel("Residuo", fontsize=10)
    ax.set_ylabel("Densidad", fontsize=10)
    ax.set_title(
        f"Distribucion del Residuo vs Curva Normal Teorica\n"
        f"Jarque-Bera: stat={jb_stat:.2f}, p={jb_p:.4f} | "
        f"ADF: p={adf_p:.6f}",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta_hist = OUTPUT_DIR / "ej4_histograma_ruido.png"
    fig.savefig(ruta_hist, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {ruta_hist.name}")

    # ── Guardar analisis.txt ─────────────────────────────────────────────────
    ruta_txt = OUTPUT_DIR / "ej4_analisis.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("Analisis del Residuo — Ejercicio 4\n")
        f.write("=" * 55 + "\n\n")

        f.write("ESTADISTICOS DESCRIPTIVOS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  N observaciones validas : {n_obs:,}\n")
        f.write(f"  Media                   : {media:.6f}  (ideal: 0)\n")
        f.write(f"  Desviacion tipica       : {std:.4f}   (sigma teorico: 3.5)\n")
        f.write(f"  Asimetria (skewness)    : {asimetria:.4f}  (ideal: 0)\n")
        f.write(f"  Curtosis en exceso      : {curtosis:.4f}  (ideal: 0)\n\n")

        f.write("TEST DE NORMALIDAD — Jarque-Bera\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Estadistico : {jb_stat:.4f}\n")
        f.write(f"  p-value     : {jb_p:.6f}\n")
        f.write(f"  H0          : los residuos siguen una distribucion normal\n")
        f.write(f"  Conclusion  : {'No se rechaza H0' if jb_p > 0.05 else 'Se rechaza H0'} "
                f"(alpha=0.05)\n\n")

        f.write("TEST DE ESTACIONARIEDAD — ADF\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Estadistico ADF : {adf_stat:.4f}\n")
        f.write(f"  p-value         : {adf_p:.6f}\n")
        f.write(f"  H0              : existe raiz unitaria (no estacionario)\n")
        f.write(f"  Conclusion      : {'Se rechaza H0 -> estacionario' if adf_p < 0.05 else 'No se rechaza H0 -> no estacionario'} "
                f"(alpha=0.05)\n\n")

        f.write("INTERPRETACION GENERAL\n")
        f.write("-" * 40 + "\n")
        f.write(f"  El residuo presenta media={media:.4f} (cercana a 0) y std={std:.4f}\n")
        f.write(f"  (el sigma teorico conocido es 3.5).\n")
        es_normal   = "si" if jb_p > 0.05 else "no"
        es_estacion = "si" if adf_p < 0.05 else "no"
        f.write(f"  Normalidad (JB)    : {es_normal} cumple\n")
        f.write(f"  Estacionariedad(ADF): {es_estacion} cumple\n")
        f.write(f"  Ver ACF/PACF para autocorrelacion residual.\n")

    print(f"  Guardado: {ruta_txt.name}")


# =============================================================================
# MAIN — NO MODIFICAR ESTE BLOQUE
# =============================================================================

if __name__ == "__main__":

    print("=" * 55)
    print("EJERCICIO 4 — Analisis de Series Temporales")
    print("=" * 55)

    # Paso 1: Generar la serie (NO modificar la semilla)
    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print(f"\nSerie generada:")
    print(f"  Periodo:       {serie.index[0].date()} -> {serie.index[-1].date()}")
    print(f"  Observaciones: {len(serie)}")
    print(f"  Media:         {serie.mean():.2f}")
    print(f"  Std:           {serie.std():.2f}")
    print(f"  Min / Max:     {serie.min():.2f} / {serie.max():.2f}")

    # Paso 2: Visualizar la serie completa
    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    # Paso 3: Descomponer
    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    # Paso 4: Analizar el residuo
    print("[3/3] Analizando el residuo...")
    if resultado is not None:
        analizar_residuo(resultado.resid)

    # Checklist de salidas
    print("\nSalidas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        existe = (OUTPUT_DIR / s).exists()
        estado = "OK" if existe else "FALTA"
        print(f"  [{estado}] {s}")

    print("\nEjercicio 4 completado.")
