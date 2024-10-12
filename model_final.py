from collections import defaultdict, deque
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def train(input_vector_path):
    min_distance = 15.6
    near_point_count = 30

    # Cargar datos
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    all_points = data.values

    # Crear NearestNeighbors para búsqueda rápida de vecinos
    neighbors_model = NearestNeighbors(radius=min_distance, algorithm='auto', metric='euclidean', n_jobs=3)
    neighbors_model.fit(all_points)

    # Precalcular vecinos para todos los puntos
    neighbors = neighbors_model.radius_neighbors(all_points, return_distance=False)
    near_points_count = np.array([len(n) for n in neighbors])

    # Inicialmente, todas las muestras son ruido.
    labels = np.full(all_points.shape[0], -1, dtype=np.intp)
    # Una lista de todos los puntos centrales encontrados.
    core_points = np.asarray(near_points_count >= near_point_count, dtype=np.uint8)

    label_num = 0  # Contador de etiquetas
    for i in tqdm(range(labels.shape[0])):
        if labels[i] != -1 or not core_points[i]:
            continue  # Saltar si ya etiquetado o no es núcleo

        # Comenzar una nueva etiqueta y la búsqueda en profundidad
        labels[i] = label_num
        stack = deque([i])  # Usar deque para eficiencia

        while stack:
            current_point = stack.pop()

            # Procesar los vecinos del punto actual
            for neighbor in neighbors[current_point]:
                if labels[neighbor] == -1:  # Si es ruido
                    labels[neighbor] = label_num  # Etiquetar como parte del cluster
                    if core_points[neighbor]:  # Si es un punto núcleo
                        stack.append(neighbor)  # Agregar a la pila para procesar

        label_num += 1  # Incrementar el contador de etiquetas para el siguiente cluster

    return labels, all_points


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
            color = plt.cm.get_cmap('tab10')(cluster_id % 10)  # Colores para clusters
            label = f'Cluster {cluster_id}'

        mask = (clusters == cluster_id)
        plt.scatter(all_points_2d[mask, 0], all_points_2d[mask, 1], s=50, c=[color], label=label, alpha=0.6)

    plt.title("Clusters Visualized")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
