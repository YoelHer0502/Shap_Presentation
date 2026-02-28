"""
============================================================
  SHAP - SHapley Additive exPlanations
  Ejemplo Práctico Aplicado a Colombia
  Predicción del Índice de Pobreza Multidimensional (IPM)
  por Departamento Colombiano
============================================================
  Estudiante : Ian Yoel Hernández Pérez
  Docente    : Alberto Acosta
  Universidad: Universidad Distrital Francisco José de Caldas
  Año        : 2026
============================================================
"""

# ============================================================
# PASO 1: Importación de librerías
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  SHAP - Ejemplo Colombia: IPM por Departamento")
print("=" * 60)
print()

# ============================================================
# PASO 2: Creación del conjunto de datos sintético
# ============================================================
print("PASO 2: Generando dataset basado en estadísticas del DANE...")

np.random.seed(42)

departamentos = [
    "Bogotá D.C.", "Antioquia", "Valle del Cauca", "Cundinamarca",
    "Atlántico", "Santander", "Bolívar", "Nariño", "Córdoba",
    "Tolima", "Norte de Santander", "Cauca", "Boyacá", "Magdalena",
    "Cesar", "Risaralda", "Meta", "Sucre", "Huila", "Caldas",
    "La Guajira", "Chocó", "Caquetá", "Putumayo", "Amazonas",
    "Vichada", "Guainía", "Vaupés", "Quindío", "San Andrés"
]

data = {
    "departamento": departamentos,
    "cobertura_educacion": [
        92.1, 85.3, 83.7, 80.2, 88.4, 84.1, 72.3, 68.5, 65.2,
        75.8, 78.2, 62.1, 77.4, 69.8, 71.2, 86.3, 79.5, 67.3,
        76.1, 84.7, 55.3, 48.2, 64.5, 61.8, 52.4, 40.2, 38.5,
        35.8, 88.9, 85.2
    ],
    "acceso_agua_potable": [
        99.2, 93.5, 91.2, 88.3, 95.1, 90.4, 71.2, 65.3, 62.5,
        80.1, 82.3, 60.4, 83.2, 68.7, 70.5, 92.4, 85.3, 65.2,
        78.9, 91.3, 45.2, 38.5, 65.3, 62.1, 50.3, 35.2, 32.4,
        28.7, 93.5, 94.2
    ],
    "informalidad_laboral": [
        41.2, 52.3, 55.1, 60.3, 53.2, 54.8, 72.3, 78.5, 80.2,
        65.4, 63.2, 82.1, 62.3, 74.5, 72.1, 50.2, 58.3, 76.4,
        66.2, 51.3, 85.3, 90.2, 74.5, 78.3, 82.4, 88.5, 91.3,
        93.2, 47.3, 45.8
    ],
    "hacinamiento": [
        8.2,  12.3, 13.5, 15.2, 11.3, 10.4, 22.3, 25.4, 27.8,
        18.5, 17.2, 28.9, 15.3, 23.4, 21.5, 9.3,  14.2, 24.5,
        17.8, 10.2, 32.5, 38.2, 21.3, 24.5, 29.4, 35.2, 38.5,
        42.3, 8.9,  7.8
    ],
    "analfabetismo": [
        2.1,  5.3,  5.8,  7.2,  4.5,  5.1,  13.2, 15.4, 17.8,
        9.8,  8.9,  18.2, 7.5,  14.2, 13.1, 4.2,  7.8,  15.3,
        9.2,  5.1,  24.5, 30.2, 12.5, 15.8, 20.3, 28.5, 32.4,
        36.8, 3.8,  3.2
    ],
    "cobertura_salud": [
        95.3, 90.2, 88.5, 85.3, 91.2, 89.4, 78.5, 72.3, 70.5,
        83.2, 84.5, 68.9, 84.8, 74.5, 76.2, 91.5, 86.3, 71.2,
        82.5, 90.8, 62.3, 55.4, 74.3, 70.8, 60.5, 52.3, 48.5,
        44.2, 92.3, 93.5
    ],
    "sin_alcantarillado": [
        2.5,  15.3, 17.8, 22.5, 8.3,  13.2, 38.5, 42.3, 45.8,
        25.4, 23.1, 48.2, 20.3, 40.5, 38.2, 10.3, 18.5, 42.8,
        24.5, 12.8, 55.3, 65.2, 38.5, 42.3, 50.5, 60.3, 65.8,
        70.2, 8.5,  6.2
    ],
    "ipm_porcentaje": [
        4.7,  12.8, 14.2, 17.5, 10.3, 12.1, 31.5, 35.8, 39.2,
        22.4, 20.8, 41.5, 19.2, 33.5, 30.8, 9.8,  16.2, 36.4,
        21.5, 11.2, 52.3, 63.5, 29.8, 34.5, 45.3, 58.2, 64.5,
        70.3, 8.3,  7.5
    ]
}

df = pd.DataFrame(data)
print(f"  Dataset creado: {len(df)} departamentos, {len(df.columns)-1} variables")
print()
print(df[["departamento", "ipm_porcentaje"]].to_string(index=False))
print()

# ============================================================
# PASO 3: Preparación de los datos
# ============================================================
print("PASO 3: Preparando variables de entrada y salida...")

feature_names = [
    "cobertura_educacion",
    "acceso_agua_potable",
    "informalidad_laboral",
    "hacinamiento",
    "analfabetismo",
    "cobertura_salud",
    "sin_alcantarillado"
]

X = df[feature_names].copy()
y = df["ipm_porcentaje"].copy()

print(f"  Variables de entrada (features): {feature_names}")
print(f"  Variable objetivo (target): ipm_porcentaje")
print(f"  Shape X: {X.shape}, Shape y: {y.shape}")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")
print()

# ============================================================
# PASO 4: Entrenamiento del modelo
# ============================================================
print("PASO 4: Entrenando modelo Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_samples_split=2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))

print(f"  RMSE: {rmse:.4f}")
print(f"  R²  : {r2:.4f}")
print()

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"  R² Cross-validation (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print()

# ============================================================
# PASO 5: Cálculo de valores SHAP
# ============================================================
print("PASO 5: Calculando valores SHAP con TreeExplainer...")

explainer = shap.TreeExplainer(model)
shap_values_all  = explainer.shap_values(X)
shap_values_test = explainer.shap_values(X_test)
expected_value   = float(explainer.expected_value)  # ← FIX: float() evita TypeError

print(f"  Valor base (predicción media): {expected_value:.2f}%")
print(f"  Shape valores SHAP: {shap_values_all.shape}")
print()

# ============================================================
# PASO 6: Importancia global (SHAP)
# ============================================================
print("PASO 6: Calculando importancia global de características...")

mean_abs_shap = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values_all).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("  Importancia global (media valor absoluto SHAP):")
print(mean_abs_shap.to_string(index=False))
print()

# ============================================================
# PASO 7: Análisis local - departamentos extremos
# ============================================================
print("PASO 7: Análisis local por departamentos...")

idx_max = df["ipm_porcentaje"].idxmax()
idx_min = df["ipm_porcentaje"].idxmin()

dept_max = str(df.loc[idx_max, "departamento"])      # ← FIX: str()
dept_min = str(df.loc[idx_min, "departamento"])
ipm_max  = float(df.loc[idx_max, "ipm_porcentaje"])  # ← FIX: float()
ipm_min  = float(df.loc[idx_min, "ipm_porcentaje"])

print(f"  Departamento con MAYOR IPM: {dept_max} ({ipm_max:.1f}%)")
print(f"  Departamento con MENOR IPM: {dept_min}  ({ipm_min:.1f}%)")
print()

shap_max = shap_values_all[idx_max]
shap_min = shap_values_all[idx_min]

print(f"  Contribuciones SHAP — {dept_max}:")
for feat, val in sorted(zip(feature_names, shap_max), key=lambda x: abs(x[1]), reverse=True):
    signo = "▲" if val > 0 else "▼"
    print(f"    {signo} {feat:30s}: {float(val):+.4f}")  # ← FIX: float()

print()
print(f"  Contribuciones SHAP — {dept_min}:")
for feat, val in sorted(zip(feature_names, shap_min), key=lambda x: abs(x[1]), reverse=True):
    signo = "▲" if val > 0 else "▼"
    print(f"    {signo} {feat:30s}: {float(val):+.4f}")  # ← FIX: float()

print()

# ============================================================
# PASO 8: Predicción para un nuevo departamento (ficticio)
# ============================================================
print("PASO 8: Predicción para un departamento hipotético...")

nuevo_dpto = pd.DataFrame([{
    "cobertura_educacion":  75.0,
    "acceso_agua_potable":  70.0,
    "informalidad_laboral": 68.0,
    "hacinamiento":         20.0,
    "analfabetismo":        10.5,
    "cobertura_salud":      80.0,
    "sin_alcantarillado":   30.0
}])

pred_nuevo = float(model.predict(nuevo_dpto)[0])    # ← FIX: float()
shap_nuevo = explainer.shap_values(nuevo_dpto)[0]

print(f"  Predicción IPM nuevo departamento: {pred_nuevo:.2f}%")
print()
print("  Desglose SHAP (contribución de cada variable):")
print(f"  {'Característica':<30} {'Valor':<12} {'SHAP':>8}")
print("  " + "-" * 52)
for feat, val_dato, shap_val in zip(feature_names, nuevo_dpto.iloc[0], shap_nuevo):
    signo = "▲" if shap_val > 0 else "▼"
    print(f"  {signo} {feat:<30} {float(val_dato):<12.1f} {float(shap_val):>+8.4f}")  # ← FIX: float()
print()
print(f"  Valor base: {expected_value:.4f}")
print(f"  Suma SHAP:  {float(shap_nuevo.sum()):.4f}")
print(f"  Predicción: {expected_value + float(shap_nuevo.sum()):.4f}  ≈  {pred_nuevo:.2f}%")
print()

# ============================================================
# PASO 9: Generación de visualizaciones
# ============================================================
print("PASO 9: Generando gráficas...")

# 9.1 Bar plot de importancia global
plt.figure(figsize=(10, 6))
feat_sorted = mean_abs_shap.sort_values("mean_abs_shap")
colors = ["#FF6B35" if v > feat_sorted["mean_abs_shap"].mean() else "#2B7BB9"
          for v in feat_sorted["mean_abs_shap"]]
plt.barh(feat_sorted["feature"], feat_sorted["mean_abs_shap"], color=colors)
plt.xlabel("Media del |Valor SHAP|", fontsize=12)
plt.title("Importancia Global de Características — SHAP\nPredicción del IPM Colombia",
          fontsize=13, fontweight='bold')
plt.axvline(feat_sorted["mean_abs_shap"].mean(), color='gray', linestyle='--',
            alpha=0.7, label='Media')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("shap_importancia_global.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → shap_importancia_global.png guardado")

# 9.2 Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_all, X, feature_names=feature_names,
                  plot_type="dot", show=False, max_display=7)
plt.title("SHAP Summary Plot — IPM Colombia", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → shap_summary_plot.png guardado")

# 9.3 Waterfall manual
fig, ax = plt.subplots(figsize=(10, 6))
shap_sorted_idx = np.argsort(np.abs(shap_max))[::-1]
features_sorted = [feature_names[i] for i in shap_sorted_idx]
values_sorted   = shap_max[shap_sorted_idx]
bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in values_sorted]
ax.barh(features_sorted, values_sorted, color=bar_colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Valor SHAP (contribución al IPM)", fontsize=11)
ax.set_title(
    f"SHAP Waterfall — {dept_max}\n"
    f"(IPM real: {ipm_max:.1f}%, predicho: {float(model.predict(X.iloc[[idx_max]])[0]):.1f}%)",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig("shap_waterfall_vaupes.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  → shap_waterfall_vaupes.png guardado ({dept_max})")

# 9.4 Scatter real vs predicho por región
plt.figure(figsize=(9, 5))
colores_regiones = {
    "Caribe":    "#E74C3C", "Andina":    "#2ECC71",
    "Pacífica":  "#3498DB", "Orinoquia": "#F39C12",
    "Amazonía":  "#9B59B6", "Insular":   "#1ABC9C"
}
region_map = {
    "Bogotá D.C.": "Andina", "Antioquia": "Andina", "Valle del Cauca": "Pacífica",
    "Cundinamarca": "Andina", "Atlántico": "Caribe", "Santander": "Andina",
    "Bolívar": "Caribe", "Nariño": "Pacífica", "Córdoba": "Caribe",
    "Tolima": "Andina", "Norte de Santander": "Andina", "Cauca": "Pacífica",
    "Boyacá": "Andina", "Magdalena": "Caribe", "Cesar": "Caribe",
    "Risaralda": "Andina", "Meta": "Orinoquia", "Sucre": "Caribe",
    "Huila": "Andina", "Caldas": "Andina", "La Guajira": "Caribe",
    "Chocó": "Pacífica", "Caquetá": "Amazonía", "Putumayo": "Amazonía",
    "Amazonas": "Amazonía", "Vichada": "Orinoquia", "Guainía": "Amazonía",
    "Vaupés": "Amazonía", "Quindío": "Andina", "San Andrés": "Insular"
}
for region, color in colores_regiones.items():
    idx_r = [i for i, d in enumerate(departamentos) if region_map.get(d) == region]
    if idx_r:
        plt.scatter(
            df.loc[idx_r, "ipm_porcentaje"],
            model.predict(X.iloc[idx_r]),
            label=region, color=color, s=80, alpha=0.8, edgecolors='k', linewidths=0.5
        )
for i, dept in enumerate(departamentos):
    plt.annotate(dept[:6],
                 (float(df.loc[i, "ipm_porcentaje"]),
                  float(model.predict(X.iloc[[i]])[0])),
                 fontsize=6, ha='center', va='bottom', color='gray')
lims = [float(min(y.min(), y_pred.min())) - 2,
        float(max(y.max(), model.predict(X).max())) + 2]
plt.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label="Predicción perfecta")
plt.xlabel("IPM Real (%)", fontsize=11)
plt.ylabel("IPM Predicho (%)", fontsize=11)
plt.title("Real vs Predicho — IPM Colombia por Región", fontsize=12, fontweight='bold')
plt.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.savefig("shap_real_vs_predicho.png", dpi=150, bbox_inches='tight')
plt.close()
print("  → shap_real_vs_predicho.png guardado")

# ============================================================
# PASO 10: Conclusiones
# ============================================================
print()
print("=" * 60)
print("  CONCLUSIONES DEL ANÁLISIS SHAP - COLOMBIA")
print("=" * 60)
print()
top3 = mean_abs_shap.head(3)["feature"].tolist()
print("  Las 3 variables más influyentes en el IPM son:")
for i, feat in enumerate(top3, 1):
    val = float(mean_abs_shap[mean_abs_shap["feature"] == feat]["mean_abs_shap"].values[0])
    print(f"    {i}. {feat} (SHAP medio: {val:.4f})")
print()
print(f"  Departamento con mayor vulnerabilidad: {dept_max} (IPM: {ipm_max:.1f}%)")
print(f"  Departamento con menor vulnerabilidad:  {dept_min} (IPM: {ipm_min:.1f}%)")
print()
print("  Imágenes generadas en el directorio de trabajo:")
print("    - shap_importancia_global.png")
print("    - shap_summary_plot.png")
print("    - shap_waterfall_vaupes.png")
print("    - shap_real_vs_predicho.png")
print()
print("  Desarrollado por: Ian Yoel Hernández Pérez")
print("  Docente: Alberto Acosta")
print("  UDFJC - 2026")
print("=" * 60)
