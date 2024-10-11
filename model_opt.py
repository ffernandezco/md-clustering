from collections import defaultdict
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KDTree
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def train(input_vector_path):
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    all_points = data.values
    n_points = len(all_points)

    # Crear NearestNeighbors para búsqueda rápida de vecinos
    neighbors_model = NearestNeighbors(radius=20, algorithm='auto', n_jobs=-1)
    neighbors_model.fit(all_points)

    visited_points = np.zeros(n_points, dtype=bool)  # Para marcar los puntos visitados
    near_points = [set() for _ in range(n_points)]  # Lista de sets para near_points
    core_points = np.zeros(n_points, dtype=bool)
    border_points = np.zeros(n_points, dtype=bool)
    non_core_points = np.zeros(n_points, dtype=bool)
    clusters = np.full(n_points, -1, dtype=np.int32)
    min_distance = 20
    near_point_count = 5
    current_cluster_id = 0

    # Identificar core points y sus vecinos
    for i in tqdm(range(n_points), desc="Identificando core points"):
        if not visited_points[i]:
            visited_points[i] = True
            indices = neighbors_model.radius_neighbors([all_points[i]], return_distance=False)[0]  # Vecinos dentro de min_distance
            near_points[i] = set(indices) - {i}  # Remover el mismo punto
            if len(near_points[i]) >= near_point_count:
                core_points[i] = True
            else:
                non_core_points[i] = True

    # Expansión de los clusters
    visited_points.fill(False)  # Limpiar para usar en la expansión de los clusters
    for i in tqdm(np.where(core_points)[0], desc="Expandiendo clusters"):
        if clusters[i] == -1:  # Si no ha sido asignado a un cluster
            current_cluster_id += 1
            expand_cluster(i, current_cluster_id, clusters, core_points, border_points, visited_points, near_points)

    print("Número de clusters:", current_cluster_id)
    print("Core Points:", np.sum(core_points))
    print("Border Points:", np.sum(border_points))
    print("Non-Core (Noise) Points:", np.sum(non_core_points))
    print("Finished")

    plot_clusters(all_points, clusters)


def expand_cluster(i, cluster_id, clusters, core_points, border_points, visited_points, near_points):
    # Asignar punto i al cluster actual
    clusters[i] = cluster_id
    # Para expandir el cluster, necesitamos visitar todos los puntos vecinos
    to_visit = list(near_points[i])

    while to_visit:
        j = to_visit.pop(0)
        if not visited_points[j]:
            visited_points[j] = True
            # Si es un core point
            if core_points[j]:
                # Agregar nuevos vecinos para visitar solo si no han sido visitados
                to_visit.extend([neighbor for neighbor in near_points[j] if not visited_points[neighbor]])
                clusters[j] = cluster_id
            else:
                # Si no es core point, pero está dentro de un cluster, es un border point
                if clusters[j] == -1:
                    clusters[j] = cluster_id
                    border_points[j] = True


def plot_clusters(all_points, clusters):
    # Reducir las dimensiones a 2D si es necesario
    if all_points.shape[1] > 2:
        pca = PCA(n_components=2)
        all_points_2d = pca.fit_transform(all_points)
    else:
        all_points_2d = all_points

    # Dibujar los puntos, coloreando según el cluster asignado
    plt.figure(figsize=(10, 7))
    unique_clusters = np.unique(clusters)

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            color = 'k'  # Ruido en color negro
            label = 'Noise'
        else:
            color = plt.cm.get_cmap('tab10')(cluster_id / len(unique_clusters))  # Colores para clusters
            label = f'Cluster {cluster_id}'

        mask = (clusters == cluster_id)
        plt.scatter(all_points_2d[mask, 0], all_points_2d[mask, 1], s=50, c=[color], label=label, alpha=0.6)

    plt.title("Clusters Visualized")
    plt.legend()
    plt.show()


# Llama a esta función después de terminar el proceso de clustering



# El cálculo de distancia ya no es necesario dado que usamos NearestNeighbors
