from collections import defaultdict
import numpy as np
import pandas as pd
import time
from tqdm import tqdm


def train(input_vector_path):
    data = pd.read_csv(input_vector_path, header=None, delimiter=',')

    all_points = data.values.tolist()
    visited_points = set()
    near_points = defaultdict(set)
    core_points = set()
    border_points = set()
    non_core_points = set()
    clusters = {}
    min_distance = 20
    near_point_count = 5
    exp = 2
    current_cluster_id = 0

    # Identificar core points y sus vecinos
    for i in tqdm(range(len(all_points)), desc="Identificando core points"):
        if i not in visited_points:
            visited_points.add(i)
            for j in range(len(all_points)):
                if i != j and i not in near_points[j] and len(near_points[i]) < near_point_count and distance(all_points[i], all_points[j], exp) < min_distance:
                    near_points[i].add(j)
                    near_points[j].add(i)
            if len(near_points[i]) >= near_point_count:
                core_points.add(i)
            else:
                non_core_points.add(i)

    # Expansión de los clusters
    visited_points.clear()  # Limpiamos para usar en la expansión de los clusters
    for i in tqdm(core_points, desc="Expandiendo clusters"):
        if i not in clusters:  # Si no ha sido asignado a un cluster
            current_cluster_id += 1
            expand_cluster(i, current_cluster_id, clusters, core_points, border_points, visited_points, near_points,
                           all_points, exp, min_distance, near_point_count)

    print("Clusters:", clusters)
    print("Core Points:", core_points)
    print("Border Points:", border_points)
    print("Non-Core (Noise) Points:", non_core_points)
    print("Finished")


def expand_cluster(i, cluster_id, clusters, core_points, border_points, visited_points, near_points, all_points, exp,
                   min_distance, near_point_count):
    # Asignar punto i al cluster actual
    clusters[i] = cluster_id
    # Para expandir el cluster, necesitamos visitar todos los puntos vecinos
    to_visit = list(near_points[i])

    while to_visit:
        j = to_visit.pop(0)
        if j not in visited_points:
            visited_points.add(j)
            # Si es un core point
            if j in core_points:
                to_visit.extend(list(set(near_points[j]) - visited_points))  # Agregar nuevos vecinos para visitar
                clusters[j] = cluster_id
            else:
                # Si no es core point, pero está dentro de un cluster, es un border point
                if j not in clusters:
                    clusters[j] = cluster_id
                    border_points.add(j)


def distance(a, b, exp=2):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError("Los vectores deben ser del mismo tamaño")

    return np.sum(np.abs(a - b) ** exp) ** (1 / exp)
