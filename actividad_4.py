import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Define rutas_posibles con datos de ejemplo
# Esto es un marcador de posición, reemplázalo con tus datos de ruta reales
rutas_posibles = [
    (['A', 'B', 'C'], 30),  # Ruta: A -> B -> C, Tiempo: 30 minutos
    (['A', 'D', 'C'], 45),  # Ruta: A -> D -> C, Tiempo: 45 minutos
    (['A', 'B', 'E', 'C'], 60),  # Ruta: A -> B -> E -> C, Tiempo: 60 minutos
]

# Dataset de ejemplo para aprendizaje no supervisado basado en el grafo de transporte
# Como no existen fuentes de datos específicas proporcionadas, se creará un dataset
# basado en las posibles rutas y sus tiempos calculados por la función buscar_ruta.

# Generar datos basados en las rutas encontradas
data = []
for ruta, tiempo in rutas_posibles:
    # Se extraen algunas características de la ruta para el clustering
    num_paradas = len(ruta)
    distancia_aproximada = tiempo  # Se asume que el tiempo es una proxy de la distancia
    data.append([num_paradas, distancia_aproximada, tiempo])

df_rutas = pd.DataFrame(data, columns=['num_paradas', 'distancia_aproximada', 'tiempo_total'])

print("Muestra del dataset generado para aprendizaje no supervisado:")
print(df_rutas.head())

# 1. Preprocesamiento de los datos
# Escalado de características para que tengan una media de 0 y una desviación estándar de 1.
# Esto es importante para algoritmos basados en la distancia como K-Means.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_rutas)
scaled_df = pd.DataFrame(scaled_features, columns=df_rutas.columns)

print("\nDatos escalados:")
print(scaled_df.head())

# 2. Desarrollo de modelos de aprendizaje no supervisado

# a) K-Means Clustering
# El objetivo es agrupar las rutas similares en función de sus características.

# Determinar el número óptimo de clusters (método del codo y silhouette score)
inertia = []
silhouette_coefficients = []
# Se cambió el rango para que comience en 2 y termine en min(11, n_samples)
# para asegurar que k siempre sea menor que el número de muestras
k_values = range(2, min(11, scaled_df.shape[0] + 1))