from collections import defaultdict, deque
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_csv(input_vector_path):
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    return data.values


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)



def compute_distances(i, all_points, block_size, min_distance):
    block_start = max(0, i - block_size)
    block_end = min(all_points.shape[0], i + block_size + 1)

    distances = np.linalg.norm(all_points[block_start:block_end] - all_points[i], axis=1)
    neighbor_indices = np.where(distances <= min_distance)[0] + block_start

    return neighbor_indices[neighbor_indices != i]  # Excluir el punto mismo

def find_neighbors(all_points, min_distance, near_point_count):
    n_points = all_points.shape[0]
    block_size = 1000  # Tamaño del bloque

    # Uso de joblib para paralelizar el cálculo
    neighbors = Parallel(n_jobs=-1)(delayed(compute_distances)(i, all_points, block_size, min_distance) for i in tqdm(range(n_points), desc="Finding neighbors"))

    return neighbors


def train(all_points, min_distance=15, near_point_count=25):
    neighbors = find_neighbors(all_points, min_distance, near_point_count)

    labels = np.full(all_points.shape[0], -1, dtype=int)  # Inicialmente todas las muestras son ruido.
    core_points = np.array([len(n) >= near_point_count for n in neighbors], dtype=bool)

    label_num = 0  # Contador de etiquetas
    for i in tqdm(range(labels.shape[0]), desc="Clustering"):
        if labels[i] != -1 or not core_points[i]:
            continue  # Saltar si ya etiquetado o no es núcleo

        labels[i] = label_num
        stack = deque([i])  # Usar deque para eficiencia

        while stack:
            current_point = stack.pop()

            for neighbor in neighbors[current_point]:
                if labels[neighbor] == -1:  # Si es ruido
                    labels[neighbor] = label_num  # Etiquetar como parte del cluster
                    if core_points[neighbor]:  # Si es un punto núcleo
                        stack.append(neighbor)  # Agregar a la pila para procesar

        label_num += 1  # Incrementar el contador de etiquetas para el siguiente cluster

    return labels


def plot_clusters(all_points, clusters, conf):
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

    plt.title("Clusters Visualized   ε:" + str(conf[0]) + " minPts:" + str(conf[1]))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def save_cluster_vectors_to_csv(all_points, clusters, max_per_cluster=10, output_csv_path="cluster_vectors.csv"):
    cluster_dict = defaultdict(list)

    # Agrupar índices por clúster, incluyendo ruido (-1)
    for idx, cluster_id in enumerate(clusters):
        cluster_dict[cluster_id].append(idx)

    # Crear una lista para almacenar los resultados
    csv_rows = []

    # Guardar hasta 'max_per_cluster' vectores por clúster
    for cluster_id, indices in cluster_dict.items():
        # Limitar la cantidad de vectores a 'max_per_cluster'
        for idx in indices[:max_per_cluster]:
            # Agregar el clúster, el índice de instancia y el vector
            row = [cluster_id, idx] + list(all_points[idx])
            csv_rows.append(row)

    # Guardar en un archivo CSV sin encabezado
    pd.DataFrame(csv_rows).to_csv(output_csv_path, index=False, header=False)
    print(f"Vectores guardados en {output_csv_path}")


def save_cluster_texts_to_csv(cluster_file_path, text_file_path, output_csv_path="cluster_texts.csv", c=4):
    # Leer el archivo que contiene los cluster IDs y los índices
    cluster_data = pd.read_csv(cluster_file_path, header=None)

    # Leer el archivo que contiene los textos (asumiendo que tiene un índice que corresponde a las instancias)
    texts = pd.read_csv(text_file_path, header=None, delimiter='|')[c].tolist()

    # Crear una lista para almacenar los resultados
    csv_rows = []

    # Iterar sobre cada fila en el archivo de clusters
    for _, row in cluster_data.iterrows():
        cluster_id = row[0]
        index = int(row[1])

        # Obtener el texto asociado al índice
        if index < len(texts):
            associated_text = texts[index]  # Asumiendo que el texto está en la primera columna
            # Agregar el clúster y el texto a la fila
            csv_rows.append([cluster_id, associated_text])

    # Guardar en un archivo CSV sin encabezado
    pd.DataFrame(csv_rows).to_csv(output_csv_path, index=False, header=False)
    print(f"Textos asociados guardados en {output_csv_path}")
