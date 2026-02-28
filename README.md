# 📊 SHAP — SHapley Additive exPlanations
### Aplicado al Índice de Pobreza Multidimensional en Colombia

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-0.44%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Activo-brightgreen)

**Estudiante:** Ian Yoel Hernández Pérez  
**Docente:** Alberto Acosta  
**Universidad Distrital Francisco José de Caldas**  
**2026**

</div>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Archivos](#-archivos)
- [Ejemplo de uso](#-ejemplo-de-uso)
- [Resultados](#-resultados)
- [Tecnologías usadas](#-tecnologías-usadas)
- [Referencias](#-referencias)

---

## 📌 Descripción

Este repositorio contiene los materiales de exposición sobre la biblioteca **SHAP** (*SHapley Additive exPlanations*), una herramienta de Python para explicar las predicciones de modelos de Machine Learning usando la teoría de valores de Shapley de la teoría de juegos cooperativos.

El ejemplo práctico central predice el **Índice de Pobreza Multidimensional (IPM)** de los 30 departamentos de Colombia usando variables socioeconómicas del **DANE (Encuesta de Calidad de Vida 2022)**, y luego aplica SHAP para explicar cuáles variables influyen más en cada predicción.

---

## 📁 Estructura del repositorio

```
Shap_Presentation/
│
├── 📓 shap_colombia.ipynb       # Notebook paso a paso con visualizaciones
├── 🐍 shap_colombia.py          # Script Python equivalente al notebook
│
├── 📄 shap_documento.pdf        # Documento LaTeX: funciones y comandos de SHAP
├── 📄 shap_documento.tex        # Fuente LaTeX del documento
│
├── 📊 shap_presentacion.pdf     # Presentación LaTeX (Beamer) — 13 diapositivas
├── 📊 shap_presentacion.tex     # Fuente LaTeX de la presentación
│
└── 📖 README.md                 # Este archivo
```

---

## ⚙️ Instalación

**1. Clonar el repositorio**

```bash
git clone https://github.com/YoelHer0502/Shap_Presentation.git
cd Shap_Presentation
```

**2. Crear entorno virtual (recomendado)**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Instalar dependencias**

```bash
pip install shap scikit-learn pandas numpy matplotlib
```

> **Versiones mínimas recomendadas:** Python ≥ 3.8, SHAP ≥ 0.44, scikit-learn ≥ 1.3

---

## 📂 Archivos

### `shap_colombia.ipynb` — Jupyter Notebook

Notebook interactivo con **14 pasos comentados**:

| Paso | Descripción |
|------|-------------|
| 1 | Importación de librerías |
| 2 | Creación del dataset (30 departamentos, 7 variables) |
| 3 | Exploración visual del IPM por departamento |
| 4 | Preparación de datos (train/test split 80/20) |
| 5 | Entrenamiento del modelo Gradient Boosting |
| 6 | Cálculo de valores SHAP con `TreeExplainer` |
| 7 | Summary Plot — importancia global |
| 8 | Waterfall Plot — explicación local por departamento |
| 9 | Force Plot — visualización interactiva |
| 10 | Dependence Plot — relación variable–SHAP |
| 11 | Beeswarm Plot — distribución completa |
| 12 | Heatmap — contribuciones por departamento |
| 13 | Predicción para nuevo departamento hipotético |
| 14 | Ranking de variables por departamento |

**Ejecutar el notebook:**
```bash
jupyter notebook shap_colombia.ipynb
```

---

### `shap_colombia.py` — Script Python

Versión ejecutable directamente desde terminal. Genera 4 gráficas PNG en el directorio de trabajo:

```bash
python shap_colombia.py
```

**Salida esperada:**
```
shap_importancia_global.png   # Bar plot de importancia SHAP
shap_summary_plot.png         # Beeswarm / dot summary
shap_waterfall_vaupes.png     # Waterfall del depto. con mayor IPM
shap_real_vs_predicho.png     # Scatter real vs predicho por región
```

---

### `shap_documento.pdf` — Documento de referencia

Documento académico en LaTeX de **9 páginas** que cubre:

- Fundamentos matemáticos del valor de Shapley
- Todos los tipos de `Explainer` disponibles en SHAP
- Funciones y parámetros principales: `summary_plot`, `waterfall_plot`, `force_plot`, `dependence_plot`, `beeswarm`, `heatmap`, `scatter`
- Tablas de compatibilidad con modelos (XGBoost, LightGBM, Random Forest, Keras, etc.)

---

### `shap_presentacion.pdf` — Presentación Beamer

Presentación de **13 diapositivas** con el siguiente orden:

1. Portada
2. Agenda
3. ¿Quién fue Lloyd Shapley?
4. Teoría de juegos aplicada a SHAP
5. ¿Por qué fue necesario crear SHAP?
6. Instalación
7. Modelos compatibles
8. Ejemplo básico con código
9. Tipos de explicadores
10. Resumen
11. Bibliografía
12. Agradecimientos
13. Link al repositorio

---

## 🚀 Ejemplo de uso

```python
import shap
from sklearn.ensemble import GradientBoostingRegressor

# Entrenar modelo
model = GradientBoostingRegressor().fit(X_train, y_train)

# Crear explicador
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizar importancia global
shap.summary_plot(shap_values, X_test)

# Explicar una predicción individual
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist()
))
```

---

## 📈 Resultados

El modelo Gradient Boosting entrenado sobre las variables del DANE logra:

| Métrica | Valor |
|---------|-------|
| RMSE (test) | ~1.8% |
| R² (test) | ~0.99 |
| R² CV-5 | ~0.98 ± 0.01 |

Las variables con **mayor influencia** en el IPM según SHAP son:

1. `analfabetismo`
2. `sin_alcantarillado`
3. `informalidad_laboral`

---

## 🛠️ Tecnologías usadas

| Herramienta | Versión | Uso |
|---|---|---|
| Python | ≥ 3.8 | Lenguaje principal |
| SHAP | ≥ 0.44 | Explicabilidad del modelo |
| scikit-learn | ≥ 1.3 | Modelo y métricas |
| pandas | ≥ 2.0 | Manejo de datos |
| numpy | ≥ 1.24 | Operaciones numéricas |
| matplotlib | ≥ 3.7 | Visualizaciones |
| LaTeX / Beamer | TeX Live 2023 | Documentos y presentación |

---

## 📚 Referencias

- Repositorio oficial SHAP: https://github.com/shap/shap
- PyPI: https://pypi.org/project/shap/
- Documentación: https://shap.readthedocs.io/
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017.
- Shapley, L. S. (1953). *A value for n-person games*. Contributions to the Theory of Games, 2(28), 307–317.
- DANE — Encuesta de Calidad de Vida 2022: https://www.dane.gov.co/

---

<div align="center">

Desarrollado por **Ian Yoel Hernández Pérez**  
Universidad Distrital Francisco José de Caldas — 2026

</div>
