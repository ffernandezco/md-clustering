from collections import defaultdict
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KDTree
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def train(input_vector_path):
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    all_points = data.values
    n_points = len(all_points)

    # Crear KDTree para búsqueda rápida de vecinos
    tree = KDTree(all_points)

    visited_points = np.zeros(n_points, dtype=bool)  # Para marcar los puntos visitados
    near_points = defaultdict(set)
    core_points = set()
    border_points = set()
    non_core_points = set()
    clusters = {}
    min_distance = 20
    near_point_count = 5
    current_cluster_id = 0

    # Identificar core points y sus vecinos
    for i in tqdm(range(n_points), desc="Identificando core points"):
        if not visited_points[i]:
            visited_points[i] = True
            indices = tree.query_radius([all_points[i]], r=min_distance)[0]  # Vecinos dentro de min_distance
            near_points[i] = set(indices) - {i}  # Remover el mismo punto
            if len(near_points[i]) >= near_point_count:
                core_points.add(i)
            else:
                non_core_points.add(i)

    # Expansión de los clusters
    visited_points.fill(False)  # Limpiar para usar en la expansión de los clusters
    for i in tqdm(core_points, desc="Expandiendo clusters"):
        if i not in clusters:  # Si no ha sido asignado a un cluster
            current_cluster_id += 1
            expand_cluster(i, current_cluster_id, clusters, core_points, border_points, visited_points, near_points)

    print("Clusters:", clusters)
    print("Core Points:", core_points)
    print("Border Points:", border_points)
    print("Non-Core (Noise) Points:", non_core_points)
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
            if j in core_points:
                # Agregar nuevos vecinos para visitar solo si no han sido visitados
                to_visit.extend([neighbor for neighbor in near_points[j] if not visited_points[neighbor]])
                clusters[j] = cluster_id
            else:
                # Si no es core point, pero está dentro de un cluster, es un border point
                if j not in clusters:
                    clusters[j] = cluster_id
                    border_points.add(j)


def plot_clusters(all_points, clusters):
    # Convertir los clusters a un array numpy para indexado fácil
    cluster_labels = np.array([clusters.get(i, -1) for i in range(len(all_points))])

    # Reducir las dimensiones a 2D si es necesario
    if all_points.shape[1] > 2:
        pca = PCA(n_components=2)
        all_points_2d = pca.fit_transform(all_points)
    else:
        all_points_2d = all_points

    # Dibujar los puntos, coloreando según el cluster asignado
    plt.figure(figsize=(10, 7))
    unique_clusters = set(cluster_labels)

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            color = 'k'  # Ruido en color negro
            label = 'Noise'
        else:
            color = plt.cm.get_cmap('tab10')(cluster_id / len(unique_clusters))  # Colores para clusters
            label = f'Cluster {cluster_id}'

        mask = (cluster_labels == cluster_id)
        plt.scatter(all_points_2d[mask, 0], all_points_2d[mask, 1], s=50, c=[color], label=label, alpha=0.6)

    plt.title("Clusters Visualized")
    plt.legend()
    plt.show()


# Llama a esta función después de terminar el proceso de clustering



# El cálculo de distancia ya no es necesario dado que usamos KDTree
