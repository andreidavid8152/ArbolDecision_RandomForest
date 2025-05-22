# Clasificación con Random Forest

#Explicación:
#División de los datos: Al igual que en el modelo anterior, el dataset se divide en un conjunto de entrenamiento y uno de prueba.
#Escalado de características: Se realiza un escalado para garantizar que las características estén en la misma escala y evitar que un valor muy grande influencie más que otros.
#Entrenamiento del modelo Random Forest: Se utiliza el clasificador RandomForestClassifier, en este caso, con 10 estimadores (árboles) y el criterio de entropía para la división.
#Predicciones: Se realiza la predicción tanto para un nuevo ejemplo como para el conjunto de prueba.
#Evaluación del modelo: Se genera la matriz de confusión y se calcula la precisión para evaluar el rendimiento del modelo.
#Visualización: Se muestran las fronteras de decisión en el conjunto de entrenamiento y prueba, lo que ayuda a entender cómo el modelo clasifica los datos.

# Importación de las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargando el dataset
dataset = pd.read_csv('Social_Network_Ads.csv')  # Lee el archivo CSV con los datos
X = dataset.iloc[:, :-1].values  # Extrae las características (Edad, Salario Estimado)
y = dataset.iloc[:, -1].values  # Extrae la variable objetivo (Compra o No Compra)

# Dividiendo el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print("Conjunto de entrenamiento (X_train):")
print(X_train)
print("Etiquetas de entrenamiento (y_train):")
print(y_train)
print("Conjunto de prueba (X_test):")
print(X_test)
print("Etiquetas de prueba (y_test):")
print(y_test)

# Escalado de características (Feature Scaling)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Ajusta y transforma el conjunto de entrenamiento
X_test = sc.transform(X_test)  # Transforma el conjunto de prueba
print("Conjunto de entrenamiento escalado (X_train):")
print(X_train)
print("Conjunto de prueba escalado (X_test):")
print(X_test)

# Entrenando el modelo de Random Forest sobre el conjunto de entrenamiento
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)  # Usamos 'entropy' como criterio
classifier.fit(X_train, y_train)  # Ajusta el modelo a los datos de entrenamiento

# Predicción de un nuevo resultado (Ejemplo: Edad = 30, Salario = 87000)
resultado = classifier.predict(sc.transform([[30, 87000]]))  # Predice si comprará o no
print(f"Predicción para edad 30 y salario 87000: {resultado}")

# Predicción sobre el conjunto de prueba
y_pred = classifier.predict(X_test)  # Predice las etiquetas para el conjunto de prueba
print("Predicciones sobre el conjunto de prueba:")
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))  # Compara las predicciones con las etiquetas reales

# Creando la Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)  # Genera la matriz de confusión
print("Matriz de Confusión:")
print(cm)
print("Precisión del modelo:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # Muestra la precisión del modelo

# Visualización de los resultados en el conjunto de entrenamiento
X_set, y_set = sc.inverse_transform(X_train), y_train  # Inversa para visualizar en la escala original
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificación con Random Forest (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba
X_set, y_set = sc.inverse_transform(X_test), y_test  # Inversa para visualizar en la escala original
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Clasificación con Random Forest (Conjunto de Prueba)')
plt.xlabel('Edad')
plt.ylabel('Salario Estimado')
plt.legend()
plt.show()
