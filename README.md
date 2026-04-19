# Práctica Final — Estadística para Data Science

**Autor:** Alejandro Pujana Quintero  
**Master:** Data Science & Inteligencia Artificial  
**Asignatura:** Estadística para Data Science  
**Fecha de entrega:** Abril 2026

---

## Descripción

Práctica final del módulo de Estadística. Cubre cuatro ejercicios de análisis y modelado
de datos: análisis descriptivo, regresión lineal con sklearn, implementación OLS desde
cero con NumPy y análisis de series temporales.

Los ejercicios 1 y 2 usan el dataset **Loan Approval Data 2025** (Kaggle, 50,000 registros).
Los ejercicios 3 y 4 usan datos sintéticos generados con semilla=42.

---

## Estructura del repositorio

```
Evolve-Estadistica-Alejandro-Pujana/
├── practica_final_Pujana_Quintero_Alejandro/
│   ├── ejercicio1_descriptivo.py       EDA completo con estadísticos y visualizaciones
│   ├── ejercicio2_inferencia.py        Regresión lineal con sklearn + summary OLS
│   ├── ejercicio3_regresion_multiple.py OLS implementado desde cero con NumPy
│   ├── ejercicio4_series_temporales.py Descomposición y análisis de serie temporal
│   ├── Respuestas.md                   Respuestas razonadas con valores numéricos reales
│   ├── data/
│   │   └── Loan_approval_data_2025.csv Dataset principal (ejercicios 1 y 2)
│   └── output/                         Gráficos y archivos generados por cada script
│       ├── ej1_*.png / .csv
│       ├── ej2_*.png / .txt
│       ├── ej3_*.png / .txt
│       └── ej4_*.png / .txt
├── notebooks/                          Cuadernos de análisis y documentación
│   ├── ej1_analisis_descriptivo.ipynb
│   ├── ej2_regresion_lineal.ipynb
│   ├── ej3_regresion_numpy.ipynb
│   └── ej4_series_temporales.ipynb
├── requirements.txt                    Dependencias del proyecto
└── README.md
```

---

## Ejercicios

- **Ejercicio 1** — Análisis estadístico descriptivo completo: estadísticos, distribuciones,
  detección de outliers (IQR), variables categóricas y matriz de correlaciones de Pearson.

- **Ejercicio 2** — Regresión lineal múltiple con sklearn: preprocesamiento (OHE, StandardScaler,
  split 80/20), métricas MAE/RMSE/R², análisis de residuos e inferencia estadística con statsmodels.

- **Ejercicio 3** — Implementación OLS desde cero con NumPy: solución analítica β = (X'X)⁻¹X'y,
  métricas sin sklearn, validación contra sklearn con diferencia=0.

- **Ejercicio 4** — Análisis de series temporales: descomposición aditiva (period=365),
  tests ADF y Jarque-Bera sobre el residuo, ACF/PACF.

---

## Ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar cada ejercicio

Desde la carpeta `practica_final_Pujana_Quintero_Alejandro/`:

```bash
python ejercicio1_descriptivo.py
python ejercicio2_inferencia.py
python ejercicio3_regresion_multiple.py
python ejercicio4_series_temporales.py
```

Cada script genera sus salidas automáticamente en `output/`.

---

## Tecnologías

- Python 3.14
- NumPy · Pandas · Matplotlib · Seaborn
- Scikit-Learn · SciPy · Statsmodels
