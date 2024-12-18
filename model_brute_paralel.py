import pickle
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from joblib import Parallel, delayed, dump, load


def read_csv(input_vector_path):
    # Cargar datos
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    return data.values


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def find_neighbors(point, all_points, min_distance, near_point_count):
    distances = np.linalg.norm(all_points - point, axis=1)
    nearest_indices = np.argsort(distances)[:near_point_count]
    return [idx for idx in nearest_indices if distances[idx] <= min_distance]


def train(all_points, min_distance=15, near_point_count=25, safe=True, output_model="model/neighbors_and_labels.pkl"):
    neighbor_matrix = {}
    num_vectors = len(all_points)
    near_points_count = np.zeros(num_vectors, dtype=int)  # Para contar los vecinos de cada punto

    # Buscar vecinos para cada punto
    for i in tqdm(range(num_vectors), desc="Buscando vecinos"):
        neighbors = []
        for j in range(num_vectors):
            if i != j:  # No comparar el vector consigo mismo
                # Calcular la distancia entre el vector i y el vector j
                distance = np.linalg.norm(all_points[i] - all_points[j])
                if distance <= min_distance:
                    neighbors.append(j)
        neighbor_matrix[i] = neighbors
        near_points_count[i] = len(neighbors)  # Contar los vecinos del punto i

    # Determinar los core points usando near_point_count
    core_points = np.asarray(near_points_count >= near_point_count - 1, dtype=np.uint8)

    # Inicialmente, todas las muestras son ruido.
    labels = np.full(all_points.shape[0], -1, dtype=np.intp)

    label_num = 0  # Contador de etiquetas
    for i in tqdm(range(labels.shape[0]), desc="Clustering"):
        if labels[i] != -1 or not core_points[i]:
            continue  # Saltar si ya etiquetado o no es núcleo

        # Comenzar una nueva etiqueta y la búsqueda en profundidad
        labels[i] = label_num
        stack = deque([i])  # Usar deque para eficiencia

        while stack:
            current_point = stack.pop()

            # Procesar los vecinos del punto actual
            for neighbor in neighbor_matrix[current_point]:
                if labels[neighbor] == -1:  # Si es ruido
                    labels[neighbor] = label_num  # Etiquetar como parte del cluster
                    if core_points[neighbor]:  # Si es un punto núcleo
                        stack.append(neighbor)  # Agregar a la pila para procesar

        label_num += 1  # Incrementar el contador de etiquetas para el siguiente cluster

    if safe:
        with open(output_model, 'wb') as f:
            dump((labels, min_distance, near_point_count), output_model)
            print(f"Modelo guardado en '{output_model}' correctamente.")

    return labels


def classify(test, input_model="model/neighbors_and_labels.pkl", n_jobs=-1):
    with open(input_model, 'rb') as f:
        labels, min_distance, near_point_count = load(input_model)
        print(f"Modelo cargado desde '{input_model}' correctamente.")

    all_points = read_csv("data/VECTOR_BERTuit90%.csv")

    # Encontrar vecinos para cada punto usando paralelización
    neighbors = Parallel(n_jobs=n_jobs)(
        delayed(find_neighbors)(point, all_points, min_distance, near_point_count + 1)
        for point in tqdm(test.values, desc="Finding neighbors")
    )

    def assign_cluster(neighbors_indices, labels):
        if neighbors_indices == -1:
            return -1  # Ruido
        # Extraemos las etiquetas de los vecinos
        neighbor_labels = labels[neighbors_indices]
        # Nos quedamos con la etiqueta mayoritaria, ignorando los -1 (ruido)
        unique_labels, counts = np.unique(neighbor_labels[neighbor_labels != -1], return_counts=True)
        if len(unique_labels) > 0:
            return unique_labels[np.argmax(counts)]  # Cluster mayoritario
        else:
            return -1  # Si no hay vecinos con etiqueta válida, también se considera ruido

    predicted_labels = [assign_cluster(n, labels) for n in neighbors]

    return predicted_labels


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
