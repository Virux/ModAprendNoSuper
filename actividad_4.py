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
