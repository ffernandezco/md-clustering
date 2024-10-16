import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import preprocess
import vectorize
import model_final

# ---- DESCOMENTAR PROCESOS QUE SE REQUIERAN ----

# PREPROCESADO
# input_file = 'data/DataI_MD.csv'  # Ruta al archivo CSV original
# output_file = 'data/DataI_MD_POST.csv'
# preprocess.preprocess(input_file, output_file)


# TOKENIZADO + VECTORIZACIÖN
# input_csv_path = 'data/DataI_MD_POST.csv'
# output_csv_path = 'data/DataI_MD_VECTOR.csv'
# # Se puede especificar el modelo (por defecto: AIDA-UPM/BERTuit-base),
# # from_tf (por defecto: True), la columna de procesado del csv de entrada (por defecto: 4)
# # y el batch_size para repartir la carga de trabajo (por defecto: 64)
# vectorize.vectorize(input_csv_path, output_csv_path, True)
preprocess.divide_csv('data/DataI_MD_VECTOR.csv', 'data/VECTOR_BERTuit90%.csv', 'data/VECTOR_test.csv', 90)
preprocess.divide_csv('data/DataI_MD_POST.csv', 'data/DataI_MD_POST90%.csv', 'data/DataI_MD_POST10%.csv', 90, delimiter="|")


# BUSCAR MEJOR CONFIGURACIÓN
instances = model_final.read_csv("data/VECTOR_BERTuit90%.csv")
external_metrics = [1, 2, 3]
best_score_silhouette = -1
best_score_davies = float('inf')
best_score_var1 = 0
best_score_var2 = 0
best_score_var3 = 0
best_configuration = {
    "silhouette": None,
    "davies": None,
    "var1": None,
    "var2": None,
    "var3": None
}

# Probar diferentes configuraciones de PCA
for n_components in [700]:  # Cambia los valores según sea necesario
    pca = PCA(n_components=n_components)
    reduced_instances = pca.fit_transform(instances)

    # Búsqueda de mejor configuración para eps y minPoints
    for eps in np.arange(10, 25, 1):
        for minPoints in range(10, 60, 10):
            for metric in ["euclidean", "manhattan", "cosine"]:
                # Entrenar el modelo y obtener las evaluaciones
                clusters, evaluation = model_final.train(reduced_instances, "data/DataI_MD_POST90%.csv", "result/" + str(n_components) + "-" + str(eps) + "-" + str(minPoints) + "-" + metric + "-evaluation.txt", eps, minPoints, metric)
                class_to_cluster_evals = evaluation[0]
                silhouette_avg = evaluation[1]
                davies_bouldin = evaluation[2]

                # Comparar con las mejores configuraciones
                if silhouette_avg is not None and silhouette_avg > best_score_silhouette:
                    best_score_silhouette = silhouette_avg
                    best_configuration["silhouette"] = {
                        'n_components': n_components,
                        'eps': eps,
                        'minPoints': minPoints,
                        'metric': metric
                    }

                if davies_bouldin is not None and davies_bouldin < best_score_davies:
                    best_score_davies = davies_bouldin
                    best_configuration["davies"] = {
                        'n_components': n_components,
                        'eps': eps,
                        'minPoints': minPoints,
                        'metric': metric
                    }

                if class_to_cluster_evals[0] > best_score_var1:
                    best_score_var1 = class_to_cluster_evals[0]
                    best_configuration["var1"] = {
                        'n_components': n_components,
                        'eps': eps,
                        'minPoints': minPoints,
                        'metric': metric
                    }

                if class_to_cluster_evals[1] > best_score_var2:
                    best_score_var2 = class_to_cluster_evals[1]
                    best_configuration["var2"] = {
                        'n_components': n_components,
                        'eps': eps,
                        'minPoints': minPoints,
                        'metric': metric
                    }

                if class_to_cluster_evals[2] > best_score_var3:
                    best_score_var3 = class_to_cluster_evals[2]
                    best_configuration["var3"] = {
                        'n_components': n_components,
                        'eps': eps,
                        'minPoints': minPoints,
                        'metric': metric
                    }

                # Guardar los resultados y gráficos
                model_final.plot_clusters(reduced_instances, clusters, [eps, minPoints, metric], "result/" + str(n_components) + "-" + str(eps) + "-" + str(minPoints) + "-" + str(metric) + "cluster_plot.png")
                model_final.save_cluster_vectors_to_csv(reduced_instances, clusters, 10, "result/" + str(n_components) + "-" + str(eps) + "-" + str(minPoints) + "-" + str(metric) + "cluster_vectors.csv")
                model_final.save_cluster_texts_to_csv("result/" + str(n_components) + "-" + str(eps) + "-" + str(minPoints) + "-" + str(metric) + "cluster_vectors.csv", "data/DataI_MD_POST.csv", "result/" + str(n_components) + "-" + str(eps) + "-" + str(minPoints) + "-" + str(metric) + "cluster_texts.csv")

# Imprimir la configuración en la consola
print("Mejores configuraciones:")
for metric, config in best_configuration.items():
    print(f"{metric}: {config}")

# Guardar la configuración en un archivo
with open("result/best_configuration.txt", "w") as f:
    f.write("Mejores configuraciones:\n")
    for metric, config in best_configuration.items():
        f.write(f"{metric}: {config}\n")


# CLASIFICAR
