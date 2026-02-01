"""
Comparacion Experimental de Clasificadores - Social Network Ads
Metricas: Precision, Recall, F1-Score, Matriz de Confusion.
Tabla resumen y visualizaciones comparativas.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score

# --- Configuracion ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

sns.set_theme(style="whitegrid")

# --- Carga de datos preprocesados y escalados ---
FEATURES = ["Gender", "Age", "EstimatedSalary"]

train_df = pd.read_csv(os.path.join(RESULTS_DIR, "train_data_scaled.csv"))
test_df = pd.read_csv(os.path.join(RESULTS_DIR, "test_data_scaled.csv"))

X_train = train_df[FEATURES].values
y_train = train_df["Purchased"]
X_test = test_df[FEATURES].values
y_test = test_df["Purchased"]

print("=" * 60)
print("COMPARACION EXPERIMENTAL DE CLASIFICADORES")
print("=" * 60)
print(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

# =============================================
# DEFINICION Y ENTRENAMIENTO DE MODELOS
# =============================================
# Modelo 1: Arbol de Decision
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# Modelo 2: SVM (GridSearchCV para ajuste de kernel y C)
print("\nEjecutando GridSearchCV para SVM...")
grid_svm = GridSearchCV(
    SVC(random_state=42),
    {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 10, 100]},
    cv=5, scoring="f1", n_jobs=-1
)
grid_svm.fit(X_train, y_train)
svm_best = grid_svm.best_estimator_
print(f"SVM - Mejores hiperparametros: {grid_svm.best_params_}")

# Modelo 3: Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

modelos = {
    "Arbol de Decision": (dt, dt.predict(X_test)),
    "SVM": (svm_best, svm_best.predict(X_test)),
    "Random Forest": (rf, rf.predict(X_test))
}

# =============================================
# 1. TABLA RESUMEN DE METRICAS
# =============================================
print("\n" + "=" * 60)
print("1. TABLA RESUMEN DE METRICAS")
print("=" * 60)

resultados = []
for nombre, (modelo, y_pred) in modelos.items():
    resultados.append({
        "Modelo": nombre,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

tabla = pd.DataFrame(resultados).set_index("Modelo")
print(f"\n{tabla.round(4).to_string()}")

# Guardar tabla como CSV
tabla_path = os.path.join(RESULTS_DIR, "16_tabla_resumen_metricas.csv")
tabla.round(4).to_csv(tabla_path)
print(f"\n  -> {tabla_path}")

# =============================================
# 2. REPORTE DE CLASIFICACION POR MODELO
# =============================================
print("\n" + "=" * 60)
print("2. REPORTE DE CLASIFICACION POR MODELO")
print("=" * 60)

for nombre, (modelo, y_pred) in modelos.items():
    print(f"\n--- {nombre} ---")
    print(classification_report(y_test, y_pred, target_names=["No Compra", "Compra"]))

# =============================================
# 3. MATRICES DE CONFUSION (comparativa)
# =============================================
print("\n" + "=" * 60)
print("3. MATRICES DE CONFUSION (comparativa)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (nombre, (modelo, y_pred)) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Compra", "Compra"],
                yticklabels=["No Compra", "Compra"], ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Valor Real")
    ax.set_title(nombre)

plt.suptitle("Matrices de Confusion - Comparacion de Modelos", fontsize=14, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "16_matrices_confusion_comparativa.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  -> {cm_path}")

# =============================================
# 4. BARPLOT COMPARATIVO DE METRICAS
# =============================================
print("\n" + "=" * 60)
print("4. BARPLOT COMPARATIVO DE METRICAS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

tabla_plot = tabla.reset_index()
metricas = ["Precision", "Recall", "F1-Score"]
x = np.arange(len(metricas))
width = 0.25
colores = ["#3498db", "#e74c3c", "#2ecc71"]

for i, (_, row) in enumerate(tabla_plot.iterrows()):
    valores = [row[m] for m in metricas]
    bars = ax.bar(x + i * width, valores, width,
                  label=row["Modelo"], color=colores[i], edgecolor="black")
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Score")
ax.set_title("Comparacion de Metricas por Modelo", fontsize=13, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metricas, fontsize=11)
ax.set_ylim(0.5, 1.08)
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
bar_path = os.path.join(RESULTS_DIR, "17_barplot_comparacion_metricas.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"  -> {bar_path}")

# =============================================
# 5. HEATMAP DE METRICAS
# =============================================
print("\n" + "=" * 60)
print("5. HEATMAP DE METRICAS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(tabla, annot=True, fmt=".4f", cmap="YlGnBu",
            linewidths=0.5, vmin=0.7, vmax=1.0, ax=ax,
            annot_kws={"size": 13})
ax.set_title("Heatmap de Metricas por Modelo", fontsize=13, fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
heat_path = os.path.join(RESULTS_DIR, "18_heatmap_metricas.png")
plt.savefig(heat_path, dpi=150)
plt.close()
print(f"  -> {heat_path}")

# =============================================
# 6. VALIDACION CRUZADA (5-fold)
# =============================================
print("\n" + "=" * 60)
print("6. VALIDACION CRUZADA (5-fold)")
print("=" * 60)

# Reunir todos los datos para CV
X_all = np.vstack([X_train, X_test])
y_all = pd.concat([y_train, y_test]).values

modelos_cv = {
    "Arbol de Decision": DecisionTreeClassifier(max_depth=4, random_state=42),
    "SVM": SVC(**grid_svm.best_params_, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
}

cv_resultados = []
for nombre, modelo in modelos_cv.items():
    scores = cross_val_score(modelo, X_all, y_all, cv=5, scoring="f1")
    cv_resultados.append({
        "Modelo": nombre,
        "F1 Media (CV)": scores.mean(),
        "F1 Desv. Estandar": scores.std(),
        "Folds": scores
    })
    print(f"  {nombre}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Boxplot de CV
fig, ax = plt.subplots(figsize=(9, 5))
cv_data = []
for res in cv_resultados:
    for score in res["Folds"]:
        cv_data.append({"Modelo": res["Modelo"], "F1-Score": score})

cv_df = pd.DataFrame(cv_data)
sns.boxplot(data=cv_df, x="Modelo", y="F1-Score",
            hue="Modelo", palette=colores, legend=False, ax=ax)
sns.stripplot(data=cv_df, x="Modelo", y="F1-Score",
              color="black", size=6, alpha=0.7, ax=ax)
ax.set_title("Validacion Cruzada (5-Fold) - F1-Score por Modelo", fontsize=13, fontweight="bold")
ax.set_ylim(0.5, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
cv_path = os.path.join(RESULTS_DIR, "19_validacion_cruzada_boxplot.png")
plt.savefig(cv_path, dpi=150)
plt.close()
print(f"  -> {cv_path}")

# =============================================
# 7. CONCLUSION
# =============================================
print("\n" + "=" * 60)
print("7. CONCLUSION")
print("=" * 60)

mejor_test = tabla["F1-Score"].idxmax()
mejor_cv = max(cv_resultados, key=lambda r: r["F1 Media (CV)"])

print(f"\n  Mejor modelo en test (F1-Score):        {mejor_test} ({tabla.loc[mejor_test, 'F1-Score']:.4f})")
print(f"  Mejor modelo en CV  (F1 Media):          {mejor_cv['Modelo']} ({mejor_cv['F1 Media (CV)']:.4f})")

print("\n" + "=" * 60)
print("COMPARACION EXPERIMENTAL COMPLETADA")
print("=" * 60)
