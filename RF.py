import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from matplotlib.colors import ListedColormap

# 1. Cargar el dataset
dataset = pd.read_csv("data/dataset_insurance.csv")

# 2. Separar características (X) y objetivo (y)
X = dataset.drop("CompraSeguro", axis=1)
y = dataset["CompraSeguro"]

# 3. Definir columnas numéricas y categóricas
numeric_cols = ["Edad", "IngresoAnual"]
categorical_cols = ["Genero", "NivelEstudios", "EstadoCivil"]

# 4. Pipeline de preprocesamiento + Random Forest
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first"), categorical_cols)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=0
    ))
])

# 5. División de datos y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
model.fit(X_train, y_train)

# --- Visualización (solo conjunto de prueba, Edad vs IngresoAnual) ---

# 6. Preparar variables para la gráfica
X_vis = X[["Edad", "IngresoAnual"]].values
y_vis = (y == "Sí").astype(int).values  # 1=Sí, 0=No

# 7. Dividir esos datos para visualización
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(
    X_vis, y_vis, test_size=0.25, random_state=0
)

# 8. Escalar solo Edad e IngresoAnual
sc_vis = StandardScaler().fit(X_vis_train)

# 9. Entrenar un Random Forest solo con estas dos features
clf_vis = RandomForestClassifier(
    n_estimators=10, criterion="entropy", random_state=0
)
clf_vis.fit(sc_vis.transform(X_vis_train), y_vis_train)

# 10. Crear meshgrid en escala original
x_min, x_max = X_vis_train[:, 0].min() - 1, X_vis_train[:, 0].max() + 1
y_min, y_max = X_vis_train[:, 1].min() - 5000, X_vis_train[:, 1].max() + 5000
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# 11. Predecir sobre el meshgrid
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf_vis.predict(sc_vis.transform(grid)).reshape(xx.shape)

# 12. Dibujar la frontera y los puntos de test
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.scatter(
    X_vis_test[:, 0], X_vis_test[:, 1],
    c=y_vis_test, cmap=ListedColormap(("red", "green")),
    edgecolors="k"
)
plt.title("Random Forest – Conjunto de Prueba")
plt.xlabel("Edad")
plt.ylabel("Ingreso Anual")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# --- ANÁLISIS ---

# 13. Predicción sobre el conjunto de prueba completo
y_pred = model.predict(X_test)
y_test_bin = (y_test == "Sí").astype(int)
y_pred_bin = (y_pred == "Sí").astype(int)

# 14. Cálculo de métricas
cm = confusion_matrix(y_test_bin, y_pred_bin)
acc = accuracy_score(y_test_bin, y_pred_bin)
prec = precision_score(y_test_bin, y_pred_bin)
rec = recall_score(y_test_bin, y_pred_bin)
f1 = f1_score(y_test_bin, y_pred_bin)

# 15. Conclusión
print(f"""
MODELO DE CLASIFICACIÓN RANDOM FOREST – ANÁLISIS DE RESULTADOS

Este modelo se ha desarrollado utilizando datos de clientes para predecir si comprarán un seguro,
en función de cinco variables:
  - Edad (años, numérica)
  - IngresoAnual (USD, numérica)
  - Genero (Masculino/Femenino, categórica)
  - NivelEstudios (Secundaria/Universitario/Postgrado, categórica)
  - EstadoCivil (Soltero/Casado/Divorciado, categórica)

El algoritmo empleado fue RandomForestClassifier con 10 árboles y criterio de entropía.
La variable objetivo fue binarizada: 1 para "Sí" (compra) y 0 para "No" (no compra).

Los resultados obtenidos fueron:

  • Exactitud (accuracy):        {acc:.3f}
  • Precisión (clase Sí):        {prec:.3f}
  • Recall (sensibilidad):       {rec:.3f}
  • F1-score:                    {f1:.3f}

La matriz de confusión muestra {cm[1,1]} verdaderos positivos (TP) y {cm[0,0]} verdaderos negativos (TN),
con {cm[0,1]} falsos positivos (FP) y {cm[1,0]} falsos negativos (FN).

Estos resultados indican que el modelo clasifica correctamente aproximadamente el {acc*100:.1f}% de los casos
y alcanza una sensibilidad del {rec*100:.1f}% para detectar a los clientes que realmente compran el seguro.
""")
