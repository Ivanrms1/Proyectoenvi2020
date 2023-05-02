import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import jax
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
import mariadb

# Definir la función de activación ReLU
def relu(x):
    return jnp.maximum(0, x)

# Definir la función de la red MLP con dos capas ocultas
def mlp(params, x):
    w1, b1, w2, b2, w3, b3 = params['w1'], params['b1'], params['w2'], params['b2'], params['w3'], params['b3']
    h1 = relu(jnp.dot(x, w1) + b1)
    h2 = relu(jnp.dot(h1, w2) + b2)
    return jnp.dot(h2, w3) + b3


# Definir la función de pérdida
def loss_fn(params, x, y):
    y_pred = mlp(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(y_pred), axis=1))

def custom_permutation_importance(wrapped_mlp, x, y, n_repeats=10, random_state=42):
    np.random.seed(random_state)
    x = np.array(x)
    baseline_score = np.mean(wrapped_mlp(x) == y)
    importances = np.zeros(x.shape[1])

    for col_idx in range(x.shape[1]):
        for _ in range(n_repeats):
            x_permuted = x.copy()
            x_permuted[:, col_idx] = np.random.permutation(x_permuted[:, col_idx])
            x_permuted_jax = jnp.array(x_permuted)
            permuted_score = np.mean(wrapped_mlp(x_permuted_jax) == y)
            importances[col_idx] += (baseline_score - permuted_score) / n_repeats

    return importances

# Definir la función de entrenamiento
def train_fn(params, x, y, alpha=0.01):
    grads = grad(loss_fn)(params, x, y)
    return {key: params[key] - alpha * grads[key] for key in params}

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

def wrapped_mlp(x):
    return np.argmax(mlp(params, x), axis=1)

def accuracy(params, x, y):
    preds = wrapped_mlp(x)
    return jnp.mean(preds == y)


# Imprimir la importancia de las características

"""Llamado a base de datos y selección de las columnas de salario y localidad"""
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
# create a connection cursor
cur = conn.cursor()

query = "SELECT ENT, SEXO, EDAD, P2_5, P2_8, P3_1, P3_3, P3_4 FROM conjunto_de_datos_tsdem_envi_2020"
cur.execute(query)
results = cur.fetchall()
data = np.array(results)
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

# Convertiemdp las etiquetas de las clases con one-hot
num_classes = 3
y_one_hot = jax.nn.one_hot(y, num_classes)


# Haciendo un split de la data, como sklearn no incluye la parte de validación se hace dos veces de forma que quede 20%, 70$ y 10%
x_train_temp, x_test, y_train_temp, y_test = train_test_split(normalized_data, y_one_hot, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.125, random_state=42)

# Convertir los conjuntos de datos divididos en matrices de JAX numpy
x_train = jnp.array(x_train)
x_test = jnp.array(x_test)
x_val = jnp.array(x_val)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)
y_val = jnp.array(y_val)

mlflow.start_run()

# Inicializar los parámetros del modelo
key = random.PRNGKey(0)
w1 = random.normal(key, (7, 64))
b1 = random.normal(key, (64,))
w2 = random.normal(key, (64, 32))
b2 = random.normal(key, (32,))
w3 = random.normal(key, (32, num_classes))
b3 = random.normal(key, (num_classes,))
params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

# Entrenar el modelo
iterations = 600
alpha = 0.001

mlflow.log_param("iterations", iterations)
mlflow.log_param("learning_rate", alpha)

train_losses = []
val_losses = []



for i in range(iterations):
    params = train_fn(params, x_train, y_train, alpha)
    if i % 10 == 0:
        y_val_pred = wrapped_mlp(x_val)
        val_accuracy = jnp.mean(y_val_pred == jnp.argmax(y_val, axis=1))
        print(f"Iteration {i}: Validation Accuracy = {val_accuracy:.4f}")

result = custom_permutation_importance(wrapped_mlp, x_test, y_test.argmax(axis=1), n_repeats=10, random_state=42)

feature_names = ['ENT', 'SEXO', 'EDAD', 'P2_5', 'P2_8', 'P3_1', 'P3_3']
print("Importancia de las características:")
for i, (feature_name, importance) in enumerate(zip(feature_names, result)):
    print(f"{feature_name}: {importance:.4f}")


train_acc = accuracy(params, x_train, y_train.argmax(axis=1))
val_acc = accuracy(params, x_val, y_val.argmax(axis=1))
test_acc = accuracy(params, x_test, y_test.argmax(axis=1))
train_loss = loss_fn(params, x_train, y_train)
val_loss = loss_fn(params, x_val, y_val)
test_loss = loss_fn(params, x_test, y_test)

mlflow.log_metric("train_accuracy", train_acc)
mlflow.log_metric("val_accuracy", val_acc)
mlflow.log_metric("test_accuracy", test_acc)
mlflow.log_metric("train_loss", train_loss)
mlflow.log_metric("val_loss", val_loss)
mlflow.log_metric("test_loss", test_loss)

import joblib

joblib.dump(params, "trained_params_mlp.pkl")
mlflow.log_artifact("trained_params_mlp.pkl")

mlflow.end_run()
