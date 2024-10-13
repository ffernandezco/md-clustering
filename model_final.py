from collections import defaultdict, deque
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
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


def train(all_points, min_distance=15, near_point_count=25):
    """
    Entrena el modelo de clustering utilizando el algoritmo DBSCAN.
    :param all_points: numpy.ndarray
        Matriz de puntos a ser agrupados, donde cada fila representa un vector.
    :param min_distance: float, opcional
        Distancia máxima entre dos puntos para ser considerados vecinos.
        El valor predeterminado es 15.
    :param near_point_count: int, opcional
        Cantidad mínima de puntos vecinos requeridos para considerar un punto como núcleo.
        El valor predeterminado es 25.
    :return: numpy.ndarray
        Etiquetas de los clusters asignadas a cada punto (incluyendo -1 para ruido).
    """
    # Precalcular vecinos para todos los puntos
    neighbors_model = NearestNeighbors(radius=min_distance, algorithm='auto', metric='euclidean', n_jobs=-1)
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

    return labels


def plot_clusters(all_points, clusters, conf, output_file):
    """
    Visualiza los clusters en un gráfico 2D.
    :param all_points: numpy.ndarray
        Matriz de puntos que han sido agrupados.
    :param clusters: numpy.ndarray
        Etiquetas de los clusters asignadas a cada punto.
    :param conf: tuple
        Tupla que contiene la configuración del modelo (eps, minPts).
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

    plt.title("Clusters Visualizados (con PCA=2)   ε:" + str(conf[0]) + " minPts:" + str(conf[1]))
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


def evaluate_clusters(clusters, csv_file, binary_columns, output_file, eps, min_points):
    """
    Función para evaluar la correlación entre clusters y variables binarias y guardar los resultados en un archivo.
    :param clusters: numpy.ndarray
        array con los clusters predichos
    :param csv_file: str
        ruta del archivo CSV que contiene las variables binarias
    :param binary_columns: array
        lista con los índices de las columnas binarias
    :param output_file: str
        ruta del archivo de salida para guardar los resultados
    :param eps: float
        valor de epsilon (min_distance)
    :param min_points: int
        valor de minPts (near_point_count)
    """
    # Leer el archivo CSV
    data = pd.read_csv(csv_file, header=None, delimiter=',')

    # Seleccionar las columnas de variables binarias
    binary_vars = data.iloc[:, binary_columns]

    # Inicializar variables para la evaluación general
    total_chi2 = 0
    total_p = 0
    num_vars = len(binary_columns)

    # Abrir el archivo de salida para escribir los resultados
    with open(output_file, 'w') as f:
        # Guardar la configuración actual
        f.write(f"# Configuración usada: eps={eps}, minPts={min_points}\n")

        # Calcular la tabla de contingencia entre clusters y cada variable binaria
        for column in binary_vars.columns:
            contingency_table = pd.crosstab(clusters, binary_vars[column])

            # Aplicar el test de chi-cuadrado
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            # Sumar los estadísticos para la evaluación general
            total_chi2 += chi2
            total_p += p

            # Guardar resultados individuales en el archivo
            f.write(f"Chi2_{column}={chi2:.4f}, p_{column}={p:.4f}\n")

        # Evaluación general (promedios)
        avg_chi2 = total_chi2 / num_vars
        avg_p = total_p / num_vars

        # Guardar resultados generales en el archivo
        f.write(f"\n# Evaluación general de las {num_vars} variables:\n")
        f.write(f"Avg_Chi2={avg_chi2:.4f}, Avg_p={avg_p:.4f}\n")

    print(f"Resultados guardados en {output_file} correctamente.")

    # Imprimir un ejemplo por cada cluster
    for cluster_id in set(clusters):  # Usa set para obtener los IDs de cluster únicos
        # Obtener un ejemplo del cluster
        example = data[clusters == cluster_id].iloc[0]  # Tomar el primer ejemplo del cluster
        # Obtener las variables binarias específicas
        binary_values = example[binary_columns].values
        # Imprimir el resultado
        print(f"Cluster: {cluster_id}, Instancia: {example.name}, Variables: {binary_values}")

