from collections import defaultdict, deque
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_csv(input_vector_path):
    # Cargar datos
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    return data.values


def train(all_points, min_distance=15, near_point_count=25):
    # Crear instancia de LSH
    lsh = LSH(num_hash_tables=20, num_projections=8)

    # Indexar los puntos
    lsh.index(all_points)

    # Encontrar vecinos para cada punto
    neighbors = []
    for point in tqdm(all_points, desc="Finding neighbors"):
        nearest = lsh.query(point, k=near_point_count)
        neighbors.append([idx for idx, dist in nearest if dist <= min_distance])

    # Inicialmente, todas las muestras son ruido.
    labels = np.full(all_points.shape[0], -1, dtype=np.intp)
    # Una lista de todos los puntos centrales encontrados.
    core_points = np.array([len(n) >= near_point_count for n in neighbors], dtype=bool)

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

### PRUEBAS ****************************************************************************************************************************
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class LSH:
    def __init__(self, num_hash_tables=10, num_projections=8):
        self.num_hash_tables = num_hash_tables
        self.num_projections = num_projections
        self.hash_tables = [defaultdict(list) for _ in range(num_hash_tables)]
        self.projections = None

    def _hash(self, vector):
        return ''.join(['1' if (vector @ proj).sum() > 0 else '0' for proj in self.projections])

    def index(self, vectors):
        self.vectors = vectors
        self.projections = [np.random.randn(vectors.shape[1], self.num_projections) for _ in
                            range(self.num_hash_tables)]

        for i, vector in enumerate(vectors):
            for table, proj in zip(self.hash_tables, self.projections):
                hash_key = self._hash(vector)
                table[hash_key].append(i)

    def query(self, vector, k=10):
        candidates = set()
        for table, proj in zip(self.hash_tables, self.projections):
            hash_key = self._hash(vector)
            candidates.update(table[hash_key])

        candidates = list(candidates)
        if len(candidates) == 0:
            return []
        distances = np.linalg.norm(self.vectors[candidates] - vector, axis=1)
        nearest = sorted(zip(candidates, distances), key=lambda x: x[1])[:k]
        return nearest
