from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import mlflow
import mlflow.sklearn
from jax import grad, jit, vmap
from jax import random
import numpy as np
import autograd.numpy as anp
from autograd import grad
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mariadb
from jax.scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

def gmm_log_prob(x, means, covs, weights):
    log_probs = []
    for i in range(len(means)):
        diff = x - means[i]
        diff = diff.reshape(-1, 1)  # Reshape diff to (7, 1)
        log_prob = -0.5 * anp.dot(diff.T, anp.linalg.solve(covs[i], diff))
        log_prob -= 0.5 * anp.log(anp.linalg.det(covs[i])) + x.shape[0] / 2 * anp.log(2 * anp.pi)
        log_probs.append(anp.log(weights[i]) + log_prob)
    return logsumexp(anp.stack(log_probs))


def gmm_loss(params, x):
    means, covs, weights = params
    log_probs = anp.array([gmm_log_prob(xi, means, covs, weights) for xi in x])
    return -anp.mean(log_probs)

def gmm_predict(params, X):
    means, covs, weights = params
    n_samples = X.shape[0]
    n_components = means.shape[0]

    log_probs = anp.array([gmm_log_prob(x, means, covs, weights) for x in X])
    labels = anp.argmax(log_probs, axis=1)
    return labels

def get_class_label(value):
    if value <= 13000:
        return 0  # Clase baja
    elif 13000 < value <= 77000:
        return 1  # Clase media
    else:
        return 2  # Clase alta
    
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Dibuja una elipse con una posición y covarianza dada."""
    ax = ax or plt.gca()

    # Convertir la covarianza en ejes principales
    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width = 2 * np.sqrt(covariance[0])
        height = 2 * np.sqrt(covariance[1])

    # Dibuja la elipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


#Creación de dataframe para extraer los datos en X y en Y
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

#preparar datos
X = np.column_stack([ent, sexo, edad, p2_5, p2_8, p3_1, p3_3])
ingresos = data[:, 7].astype(int)
y = np.array([get_class_label(val) for val in ingresos])

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividisión de datos con sklearn
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlflow.start_run()

#entrenar modelo
n_components = 2
gmm = GaussianMixture(n_components, random_state=42)
gmm.fit(X_train)    

# Predecir las etiquetas de clase para los x_test
labels = gmm.predict(X_test)
pca = PCA()
X_train_pca = pca.fit_transform(X_train)

# Obtener la importancia de cada característica
feature_importance = pca.explained_variance_ratio_
feature_names = ['ENT', 'SEXO', 'EDAD', 'P2_5', 'P2_8', 'P3_1', 'P3_3']

for name, importance in zip(feature_names, feature_importance):
    print(f"Característica {name}: {importance * 100:.2f}%")

# Calcular la precisión del modelo
y_test_class_labels = np.array([get_class_label(val) for val in ingresos[y_test]])
accuracy = np.sum(labels == y_test_class_labels) / len(labels) * 100

"""# Proyectar los datos de prueba en el espacio bidimensional de PCA
X_test_pca = pca.transform(X_test)
fig, ax = plt.subplots()
for i in range(gmm.n_components):
    draw_ellipse(gmm.means_[i, :2], gmm.covariances_[i, :2, :2], ax=ax, alpha=0.5)

# Graficar los datos proyectados en el espacio bidimensional de PCA
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=labels, cmap='viridis', marker='.', alpha=0.5)
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Visualización de GMM en el espacio de PCA bidimensional')
plt.show()
"""
# registro de métricas
mlflow.log_param("n_components", n_components)
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(gmm, "GaussianMixtureModel")
print(f"Precisión del modelo: {accuracy:.2f}%")


