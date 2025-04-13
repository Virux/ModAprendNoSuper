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

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)
    # Solo calcular silhouette score si k es menor que el número de muestras - 1
    if k < scaled_df.shape[0]:
        score = silhouette_score(scaled_df, kmeans.labels_)
        silhouette_coefficients.append(score)
    else:
        # Agregar un valor NaN a la lista silhouette_coefficients para indicar
        # que el silhouette score no se pudo calcular para este valor de k
        silhouette_coefficients.append(float('nan'))

# Visualización del método del codo
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, 'o-')
plt.title('Método del Codo para determinar el número óptimo de clusters')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')

# Visualización del Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_coefficients, 'o-')
plt.title('Silhouette Score para determinar el número óptimo de clusters')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# Basado en la visualización (el usuario deberá interpretar los gráficos),
# se elige un número de clusters 'n_clusters'. Para este ejemplo, asumiremos 3.
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_rutas['cluster_kmeans'] = kmeans.fit_predict(scaled_df)

print("\nDataset con etiquetas de cluster de K-Means:")
print(df_rutas.head())

# Visualización de los clusters K-Means (ejemplo con dos características)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_rutas, x='num_paradas', y='tiempo_total', hue='cluster_kmeans', palette='viridis')
plt.title(f'Clusters de Rutas (K-Means, {n_clusters} clusters)')
plt.xlabel('Número de Paradas')
plt.ylabel('Tiempo Total (minutos)')
plt.show()

# b) DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# El objetivo es encontrar grupos de alta densidad separados por regiones de baja densidad.
from sklearn.cluster import DBSCAN

# Determinar los parámetros óptimos para DBSCAN (eps y min_samples)
# Esto a menudo requiere experimentación y conocimiento del dominio.
# Se probarán algunos valores.
eps_values = [0.5, 1, 1.5]
min_samples_values = [3, 5, 7]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters_dbscan = dbscan.fit_predict(scaled_df)
        n_clusters_dbscan = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
        n_noise_dbscan = list(clusters_dbscan).count(-1)
        print(f"\nResultados de DBSCAN con eps={eps}, min_samples={min_samples}:")
        print(f"  Número de clusters encontrados: {n_clusters_dbscan}")
        print(f"  Número de puntos de ruido: {n_noise_dbscan}")

# Aplicar DBSCAN con parámetros elegidos (basado en la experimentación anterior)
eps_optimo = 1.0
min_samples_optimo = 5
dbscan = DBSCAN(eps=eps_optimo, min_samples=min_samples_optimo)
df_rutas['cluster_dbscan'] = dbscan.fit_predict(scaled_df)

print(f"\nDataset con etiquetas de cluster de DBSCAN (eps={eps_optimo}, min_samples={min_samples_optimo}):")
print(df_rutas.head())

# Visualización de los clusters DBSCAN (ejemplo con dos características)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_rutas, x='num_paradas', y='tiempo_total', hue='cluster_dbscan', palette='viridis')
plt.title(f'Clusters de Rutas (DBSCAN, eps={eps_optimo}, min_samples={min_samples_optimo})')
plt.xlabel('Número de Paradas')
plt.ylabel('Tiempo Total (minutos)')
plt.show()

# c) Análisis de Componentes Principales (PCA) para reducción de dimensionalidad (opcional para visualización)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
pca_df['cluster_kmeans'] = df_rutas['cluster_kmeans']
pca_df['cluster_dbscan'] = df_rutas['cluster_dbscan']

print("\nDataset con componentes principales:")
print(pca_df.head())

# Visualización de clusters K-Means en el espacio PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster_kmeans', palette='viridis')
plt.title(f'Clusters K-Means en el espacio PCA ({n_clusters} clusters)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# Visualización de clusters DBSCAN en el espacio PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='principal_component_1', y='principal_component_2', hue='cluster_dbscan', palette='viridis')
plt.title(f'Clusters DBSCAN en el espacio PCA (eps={eps_optimo}, min_samples={min_samples_optimo})')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()

# 3. Interpretación de los resultados (esto dependerá del contexto específico)
# Los clusters identificados por K-Means y DBSCAN representan grupos de rutas
# que comparten similitudes en términos de número de paradas y tiempo total (proxy de distancia).
#
# - Los clusters de K-Means intentan encontrar grupos esféricos basados en la distancia.
# - Los clusters de DBSCAN identifican áreas densas de puntos, lo que puede ser útil
#   para encontrar patrones de rutas congestionadas o rutas directas.
#
# El análisis de estos clusters podría ayudar a:
# - Identificar tipos de rutas (cortas y rápidas, largas con muchas paradas, etc.).
# - Planificar mejoras en el sistema de transporte.
# - Comprender mejor los patrones de movimiento.
#
# Es importante recordar que la interpretación de los clusters requiere un conocimiento
# profundo del sistema de transporte que se está modelando.