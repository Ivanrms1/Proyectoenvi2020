import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import mariadb
import mysql.connector
from matplotlib import pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split


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

#Convertir la variable objetivo en etiquetas de clases
def get_class_label(value):
    if value <= 13000:
        return 0  # Clase baja
    elif 13000 < value <= 77000:
        return 1  # Clase media
    else:
        return 2  # Clase alta

#función de activación para clasificación
def softmax_regression(params, x):
    w, b = params
    linear_output = jnp.dot(x, w) + b
    return jax.nn.softmax(linear_output, axis=-1)

def linear_regression(params, x):
    w, b = params
    return jnp.dot(x, w) + b

#función de pérdida
def loss_fn(params, x, y):
    preds = linear_regression(params, x)
    y_reshaped = jnp.reshape(y, (-1, 1))
    return jnp.mean((y_reshaped - preds) ** 2)

def normalize_data(data):
    min_values = jnp.min(data, axis=0)
    max_values = jnp.max(data, axis=0)
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data

def train_fn(params, x, y, alpha):
    grads = grad(loss_fn)(params, x, y)
    return [p - alpha * g for p, g in zip(params, grads)]

def categorical_crossentropy(params, x, y):
    preds = softmax_regression(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(preds), axis=-1))


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

# Combina las variables en una matriz
combined_data = jnp.column_stack((ent, sexo, edad, p2_5, p2_8, p3_1, p3_3))
normalized_data = normalize_data(combined_data)
ent_norm, sexo_norm, edad_norm, p2_5_norm, p2_8_norm, p3_1_norm, p3_3_norm = [normalized_data[:, i] for i in range(normalized_data.shape[1])]

# Concatenar todas las columnas para formar la matriz x
x = jnp.column_stack([ent, sexo, edad, p2_5, p2_8, p3_1, p3_3])
y = jnp.array([get_class_label(val) for val in data[:, 7]])

# Convertiendp las etiquetas de clase en codificación one-hot
num_classes = 3
y_one_hot = jax.nn.one_hot(y, num_classes)

# Haciendo un split de la data, como sklearn no incluye la parte de validación se hace dos veces de forma que quede 20%, 70$ y 10%
x_train_temp, x_test, y_train_temp, y_test = train_test_split(normalized_data, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.125, random_state=42)

# Convertir a matrices de JAX numpy
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)
y_val = jnp.array(y_val)
y_train_one_hot = jax.nn.one_hot(y_train, num_classes)
y_val_one_hot = jax.nn.one_hot(y_val, num_classes)

mlflow.start_run()

"""
Inicialización del modelo de regresion
"""

# Inicializar previa del modelo, con pesos y bias aleatorias
key = random.PRNGKey(0)
# Actualizar la inicialización del modelo para incluir el número de clases
params = [random.normal(key, (x.shape[1], num_classes)), random.normal(key, (num_classes,))]
# Entrenar el modelo
iterations = 30000
alpha = 0.001

mlflow.log_param("iterations", iterations)
mlflow.log_param("learning_rate", alpha)

train_losses = []
val_losses = []

for i in range(iterations):
    params = train_fn(params, x_train, y_train, alpha)
    train_loss = categorical_crossentropy(params, x_train, y_train_one_hot)
    val_loss = categorical_crossentropy(params, x_val, y_val_one_hot)
    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))


val_loss = loss_fn(params, x_val, y_val)
mlflow.log_metric("val_loss", float(val_loss))

test_loss = loss_fn(params, x_test, y_test)
mlflow.log_metric("test_loss", float(test_loss))

#for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
#    print(f"Iteración {i + 1}: MSE de entrenamiento = {train_loss}, MSE de validación = {val_loss}")


#weight_magnitudes = jnp.abs(params[0].flatten())
feature_names = ['ENT', 'SEXO', 'EDAD', 'P2_5', 'P2_8', 'P3_1', 'P3_3']
weight_magnitudes = jnp.abs(params[0].flatten())[:len(feature_names)]
sorted_feature_indices = jnp.argsort(weight_magnitudes)[::-1]

print("Características ordenadas por importancia:")
for idx in sorted_feature_indices:
    if idx < len(feature_names):
        print(f"{feature_names[idx]}: {params[0][idx, 0]}")
    else:
        print(f"Índice {idx} fuera del rango de la lista de nombres de características")

"""
y_pred = linear_regression(params, x_test)
y_pred_mean = y_pred.mean(axis=1)
y_pred_class_labels = jnp.array([get_class_label(val) for val in y_pred_mean])
mse = jnp.mean((y_test - y_pred_mean) ** 2)
print(f"Error cuadrático medio en el conjunto de prueba: {mse:.2f}")
"""

# Obtener las probabilidades de clase utilizando softmax_regression
y_pred_probs = softmax_regression(params, x_test)
y_pred_class_labels = jnp.argmax(y_pred_probs, axis=1)

# Calcular la precisión de la clasificación
accuracy = jnp.mean(y_pred_class_labels == y_test)
print(f"Classification accuracy: {accuracy * 100:.2f}%")
mlflow.log_metric("accuracy", float(accuracy))


"""plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Entrenamiento")
plt.plot(val_losses, label="Validación")
plt.xlabel("Iteración")
plt.ylabel("Error cuadrático medio (MSE)")
plt.title("MSE en cada iteración")
plt.legend()
plt.show()"""

import joblib

joblib.dump(params, "trained_params.pkl")
mlflow.log_artifact("trained_params.pkl")

mlflow.end_run()