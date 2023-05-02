import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import mariadb
from matplotlib import pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score



"""Función para conectar a mariaDB"""

#conectando al database
config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Password123!',
    'database': 'ENV'
}

# connection for MariaDB
try:
    conn = mariadb.connect(**config)

except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Enable Auto-Commit
conn.autocommit = True
cur = conn.cursor()

#Convertir la variable objetivo en etiquetas de clase
def get_class_label(value):
    if value <= 13000:
        return 0  # Clase baja
    elif 13000 < value <= 77000:
        return 1  # Clase media
    else:
        return 2  # Clase alta

def normalize_data(data):
    min_values = jnp.min(data, axis=0)
    max_values = jnp.max(data, axis=0)
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def train_and_evaluate_decision_tree(max_depth, x_train, y_train, x_val, y_val):
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_classifier.fit(x_train, y_train)
    
    y_train_pred = dt_classifier.predict(x_train)
    y_val_pred = dt_classifier.predict(x_val)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    train_mse = calculate_mse(y_train, y_train_pred)
    val_mse = calculate_mse(y_val, y_val_pred)

    return train_accuracy, val_accuracy, train_mse, val_mse


#Creación de dataframe para extraer los datos en X y en Y
query = "SELECT ENT, SEXO, EDAD, P2_5, P2_8, P3_1, P3_3, P3_4 FROM conjunto_de_datos_tsdem_envi_2020"
cur.execute(query)
results = cur.fetchall()

# Convertir los resultados en una matriz de JAX numpy

data = jnp.array(results)
ent = data[:, 0].astype(int)
sexo = data[:, 1].astype(int)
edad = data[:, 2].astype(int)
p2_5 = data[:, 3].astype(int)
p2_8 = data[:, 4].astype(int)
p3_1 = data[:, 5].astype(int)
p3_3 = data[:, 6].astype(int)


combined_data = jnp.column_stack((ent, sexo, edad, p2_5, p2_8, p3_1, p3_3))
normalized_data = normalize_data(combined_data)
ent_norm, sexo_norm, edad_norm, p2_5_norm, p2_8_norm, p3_1_norm, p3_3_norm = [normalized_data[:, i] for i in range(normalized_data.shape[1])]

# Concatenar todas las columnas para formar la matriz x
x = jnp.column_stack([ent, sexo, edad, p2_5, p2_8, p3_1, p3_3])
y = jnp.array([get_class_label(val) for val in data[:, 7]])

# 2. Convertir las etiquetas de clase en codificación one-hot
num_classes = 3
y_one_hot = jax.nn.one_hot(y, num_classes)

# Haciendo un split de la data, como sklearn no incluye la parte de validación se hace dos veces de forma que quede 20%, 70$ y 10%
x_train_temp, x_test, y_train_temp, y_test = train_test_split(normalized_data, data[:, 7], test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.125, random_state=42)

# Convertir a matrices de JAX numpy
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)
y_val = jnp.array(y_val)

mlflow.start_run()

feature_names = ['ENT', 'SEXO', 'EDAD', 'P2_5', 'P2_8', 'P3_1', 'P3_3']

#Crear y entrenar un árbol de decisión
max_depth = 6
dt_classifier = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_classifier.fit(x_train, y_train)
y_pred_class_labels = dt_classifier.predict(x_test)

#Acuraccia
accuracy = accuracy_score(y_test, y_pred_class_labels)
print(f"Classification accuracy: {accuracy * 100:.2f}%")

# Obtener las importancias de características mediante feature importances, que es la función ya definida para ello de sklearn
feature_importances = dt_classifier.feature_importances_
sorted_feature_indices = np.argsort(feature_importances)[::-1]
print("Características ordenadas por importancia (árbol de decisión):")
for idx in sorted_feature_indices:
    print(f"{feature_names[idx]}: {feature_importances[idx]}")

#plt.figure(figsize=(20, 10))
#plot_tree(dt_classifier, feature_names=feature_names, class_names=['Clase baja', 'Clase media', 'Clase alta'], filled=False, rounded=True)
#plt.show()

"""# Predecir las etiquetas de clase para los conjuntos de entrenamiento y validación
y_train_pred = dt_classifier.predict(x_train)
y_val_pred = dt_classifier.predict(x_val)

# Calcular el MSE
train_mse = calculate_mse(y_train, y_train_pred)
val_mse = calculate_mse(y_val, y_val_pred)

plt.bar(['Train', 'Validation'], [train_mse, val_mse])
plt.ylabel('MSE')
plt.title('MSE entre los datos de validación y entrenamiento')
plt.show()
"""

"""max_depths = list(range(1, 21))
train_accuracies = []
val_accuracies = []
train_mse_values = []
val_mse_values = []

for depth in max_depths:
    train_accuracy, val_accuracy, train_mse, val_mse = train_and_evaluate_decision_tree(depth, x_train_temp, y_train_temp, x_val, y_val)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    train_mse_values.append(train_mse)
    val_mse_values.append(val_mse)

plt.figure()
plt.plot(max_depths, train_accuracies, label='Train accuracy')
plt.plot(max_depths, val_accuracies, label='Validation accuracy')
plt.xlabel('Profundidad del árbol')
plt.ylabel('Precisión')
plt.title('Precisión en función de la profundidad del árbol')
plt.legend()
plt.show()"""

# Registrar métricas y parámetros en MLflow
mlflow.log_param("n_components", max_depth)
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(dt_classifier, "DT")
import joblib

joblib.dump(dt_classifier, "decision_tree_model.pkl")
mlflow.log_artifact("decision_tree_model.pkl")

mlflow.end_run()
