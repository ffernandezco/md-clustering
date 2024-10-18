from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_csv(input_vector_path):
    """
    Lee un archivo CSV y devuelve los datos como un array numpy.
    :param input_vector_path: str
        Ruta del archivo CSV que contiene los vectores de entrada.
        Se espera que el archivo no tenga encabezados y use coma como delimitador.
    :return: numpy.ndarray
        Matriz de vectores leída desde el archivo CSV.
    """
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')
    return data.values


def train(all_points, csv_file, output_file, min_distance=15, near_point_count=25, metric="euclidean"):
    """
    Entrena el modelo de clustering utilizando el algoritmo DBSCAN y calcula las métricas de calidad.
    :param all_points: numpy.ndarray
        Matriz de puntos a ser agrupados, donde cada fila representa un vector.
    :param csv_file: str
        Ruta del archivo CSV que contiene las variables binarias.
    :param output_file: str
        Ruta del archivo de salida para guardar los resultados.
    :param min_distance: float, opcional
        Distancia máxima entre dos puntos para ser considerados vecinos.
        El valor predeterminado es 15.
    :param near_point_count: int, opcional
        Cantidad mínima de puntos vecinos requeridos para considerar un punto como núcleo.
        El valor predeterminado es 25.
    :param metric: str, opcional
        Métrica de distancias
    :return: numpy.ndarray
        Etiquetas de los clusters asignadas a cada punto (incluyendo -1 para ruido).
    """
    # Precalcular vecinos para todos los puntos
    neighbors_model = NearestNeighbors(radius=min_distance, algorithm='brute', metric=metric, n_jobs=-1)
    neighbors_model.fit(all_points)
    neighbors = neighbors_model.radius_neighbors(all_points, return_distance=False)
    near_points_count = np.array([len(n) for n in neighbors])

    # Inicializaciones
    labels = np.full(all_points.shape[0], -1, dtype=np.intp)
    core_points = np.asarray(near_points_count >= near_point_count, dtype=np.uint8)

    label_num = 0  # Contador de etiquetas de clusters
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

    print("Modelo entrenado correctamente.")

    # Calcular Silhouette Index y Davies-Bouldin Index
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(all_points, labels)
        davies_bouldin = davies_bouldin_score(all_points, labels)
    else:
        silhouette_avg = None
        davies_bouldin = None

    # Evaluar clusters
    evaluation = evaluate_clusters(labels, csv_file, output_file, min_distance, near_point_count, silhouette_avg, davies_bouldin, metric)

    return labels, evaluation


def plot_clusters(all_points, clusters, conf, output_file):
    """
    Visualiza los clusters en un gráfico 2D.
    :param all_points: numpy.ndarray
        Matriz de puntos que han sido agrupados.
    :param clusters: numpy.ndarray
        Etiquetas de los clusters asignadas a cada punto.
    :param conf: triple
        Triple que contiene la configuración del modelo (eps, minPts, metric).
    :param output_file: str
        Ruta del archivo donde se guardará la visualización del gráfico.
    """
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

    plt.title("Clusters Visualizados (con PCA=2)   ε:" + str(conf[0]) + " minPts:" + str(conf[1]) + " métrica:" + str(conf[2]))
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.grid(True)

    # Guardar el plot en un archivo si se especifica
    if output_file:
        plt.savefig(output_file)
        print(f"Plot guardado en {output_file} correctamente.")

    plt.show()


def save_cluster_vectors_to_csv(all_points, clusters, max_per_cluster=10, output_csv_path="cluster_vectors.csv"):
    """
    Guarda los vectores de los clusters en un archivo CSV.
    :param all_points: numpy.ndarray
        Matriz de puntos que han sido agrupados.
    :param clusters: numpy.ndarray
        Etiquetas de los clusters asignadas a cada punto.
    :param max_per_cluster: int, opcional
        Máximo número de vectores a guardar por cluster.
        El valor predeterminado es 10.
    :param output_csv_path: str, opcional
        Ruta del archivo CSV donde se guardarán los vectores de los clusters.
        El valor predeterminado es "cluster_vectors.csv".
    """
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
    print(f"Vectores guardados en {output_csv_path} correctamente.")


def save_cluster_texts_to_csv(cluster_file_path, text_file_path, output_csv_path="cluster_texts.csv", c=4):
    """
    Guarda los textos asociados a los clusters en un archivo CSV.
    :param cluster_file_path: str
        Ruta del archivo que contiene los IDs de los clusters y los índices.
    :param text_file_path: str
        Ruta del archivo que contiene los textos (asumiendo que tiene un índice que corresponde a las instancias).
    :param output_csv_path: str, opcional
        Ruta del archivo CSV donde se guardarán los textos asociados a los clusters.
        El valor predeterminado es "cluster_texts.csv".
    :param c: int, opcional
        Índice de la columna en el archivo de texto que contiene los textos.
        El valor predeterminado es 4.
    """
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
    print(f"Textos asociados guardados en {output_csv_path} correctamente.")


def evaluate_clusters(clusters, csv_file, output_file, eps, min_points, silhouette_avg, davies_bouldin,  metric, vars=[1, 2, 3]):
    """
    Función para evaluar la correlación entre clusters y variables binarias y guardar las tablas de contingencia en un archivo.
    :param clusters: numpy.ndarray
        array con los clusters predichos
    :param csv_file: str
        ruta del archivo CSV que contiene las variables binarias
    :param output_file: str
        ruta del archivo de salida para guardar los resultados
    :param eps: float
        valor de epsilon (min_distance)
    :param min_points: int
        valor de minPts (near_point_count)
    :param silhouette_avg: float
        Valor del Silhouette Index calculado.
    :param davies_bouldin: float
        Valor del Davies-Bouldin Index calculado.
    :param metric: str
        Valor de la métrica de distancia utilizada
    :param vars: list
        Incices del archivo csv de las métricas externas
    """
    # Leer el archivo CSV
    data = pd.read_csv(csv_file, header=None, delimiter=',')

    # Seleccionar las columnas de variables binarias
    binary_vars = data.iloc[:, vars]
    class_to_cluster_evals = []

    # Abrir el archivo de salida para escribir los resultados
    with open(output_file, 'w') as f:
        # Guardar la configuración actual
        f.write(f"# Configuración usada: eps={eps}, minPts={min_points}, métrica={metric}\n")

        # Calcular la tabla de contingencia entre clusters y cada variable binaria
        for column in binary_vars.columns:
            if column == 3:  # Supongamos que la variable 3 es la columna 2
                # Modificar los valores de la columna para fusionar 1 y 2
                modified_column = binary_vars[column].replace({1: 1, 2: 1})  # Fusionar 1 y 2
                contingency_table = pd.crosstab(clusters, modified_column)
                contingency_table.index.name = "real-->"
                contingency_table.columns.name = ""
            else:
                contingency_table = pd.crosstab(clusters, binary_vars[column])
                contingency_table.index.name = "real-->"
                contingency_table.columns.name = ""

            # Calcular class-to-cluster evaluation
            total_instances = contingency_table.values.sum()
            max_per_cluster = contingency_table.max(axis=1).sum()
            accuracy = max_per_cluster / total_instances
            class_to_cluster_evals = class_to_cluster_evals.append(accuracy)

            # Guardar pureza en el archivo
            f.write(f"\n# Class-to-cluster evaluation para la variable {column}: {accuracy:.4f}\n")

            # Guardar la tabla de contingencia en el archivo
            f.write(f"\n# Tabla de contingencia para la variable {column}:\n")
            f.write(f"{contingency_table.to_string()}\n")

        # Guardar métricas de calidad
        if silhouette_avg is not None:
            f.write(f"# Silhouette Index: {silhouette_avg:.4f}\n")
        else:
            f.write("# Silhouette Index: No se puede calcular, solo un cluster.\n")

        if davies_bouldin is not None:
            f.write(f"# Davies-Bouldin Index: {davies_bouldin:.4f}\n")
        else:
            f.write("# Davies-Bouldin Index: No se puede calcular, solo un cluster.\n")

    print(f"Tablas de contingencia y métricas de calidad guardadas en {output_file} correctamente.")

    return class_to_cluster_evals, silhouette_avg, davies_bouldin
