import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import preprocess
import vectorize
import model_final

# ---- DESCOMENTAR PROCESOS QUE SE REQUIERAN ---

# PREPROCESADO
# input_file = 'data/DataI_MD.csv'  # Ruta al archivo CSV original
# output_file = 'data/DataI_MD_POST.csv'
#
# preprocess.preprocess(input_file, output_file)

# TOKENIZADO + VECTORIZACIÖN
# input_csv_path = 'data/DataI_MD_POST.csv'
# output_csv_path = 'data/DataI_MD_VECTOR.csv'

# vectorize.vectorize(input_csv_path, output_csv_path, True)
# También se puede especificar el modelo (por defecto: AIDA-UPM/BERTuit-base),
# from_tf (por defecto: True), la columna de procesado del csv de entrada (por defecto: 4)
# y el batch_size para repartir la carga de trabajo (por defecto: 64)

# ENTRENAR MODELO
# all_points = model_final.read_csv("data/VECTOR_BERTuit.csv")
# configuration = [14, 20]
# clusters = model_final.train(all_points, configuration[0], configuration[1])
# model_final.plot_clusters(all_points, clusters, configuration)
# model_final.save_cluster_vectors_to_csv(all_points, clusters, max_per_cluster=5,
#                                         output_csv_path="data/" + str(configuration[0]) + "," + str(
#                                             configuration[1]) + "cluster_vectors.csv")
# model_final.save_cluster_texts_to_csv("data/" + str(configuration[0]) + "," + str(configuration[1]) +
#                                       "cluster_vectors.csv", 'data/DataI_MD_POST.csv', "data/" +
#                                       str(configuration[0]) + "," + str(configuration[1]) + "cluster_texts.csv")

# BUSCAR MEJOR CONFIGURACION
# Leer los datos de las instancias
instances = model_final.read_csv("data/VECTOR_BERTuit90%.csv")

# Variables para almacenar la mejor configuración
best_configuration = None
best_score = -np.inf  # Inicializamos con un valor muy bajo

# Listas para almacenar los datos para el gráfico
eps_values = []
min_points_values = []
evaluacion_values = []
pca_components_values = []  # Lista para los valores de componentes PCA

# Probar diferentes configuraciones de PCA
for n_components in [50]:  # Cambia los valores según sea necesario
    pca = PCA(n_components=n_components)
    reduced_instances = pca.fit_transform(instances)

    # Búsqueda de mejor configuración para eps y minPoints
    for eps in np.arange(10, 15, 5):
        for minPoints in range(10, 25, 5):
            # Entrenar el modelo
            clusters, evaluacion = model_final.train(reduced_instances, "data/DataI_MD_POST90%.csv",
                                                     "result/" + str(eps) + "-" + str(minPoints) + "evaluation.txt",
                                                     eps, minPoints)

            # Comparar con la mejor configuración
            if evaluacion > best_score:
                best_score = evaluacion
                best_configuration = {
                    'n_components': n_components,
                    'eps': eps,
                    'minPoints': minPoints,
                    'evaluacion': evaluacion
                }

            # Almacenar los valores para el gráfico
            eps_values.append(eps)
            min_points_values.append(minPoints)
            evaluacion_values.append(evaluacion)
            pca_components_values.append(n_components)  # Agregar el número de componentes PCA

            # Guardar los resultados y gráficos
            model_final.plot_clusters(reduced_instances, clusters, [eps, minPoints],
                                      "result/" + str(eps) + "-" + str(minPoints) + "cluster_plot.png")
            model_final.save_cluster_vectors_to_csv(reduced_instances, clusters, 10,
                                                    "result/" + str(eps) + "-" + str(minPoints) + "cluster_vectors.csv")
            model_final.save_cluster_texts_to_csv("result/" + str(eps) + "-" + str(minPoints) + "cluster_vectors.csv",
                                                  "data/DataI_MD_POST.csv",
                                                  "result/" + str(eps) + "-" + str(minPoints) + "cluster_texts.csv")

# Imprimir la mejor configuración encontrada
if best_configuration:
    print("Mejor configuración encontrada:")
    print(f"Número de componentes PCA: {best_configuration['n_components']}")
    print(f"eps: {best_configuration['eps']}")
    print(f"minPoints: {best_configuration['minPoints']}")
    print(f"Evaluación: {best_configuration['evaluacion']:.4f}")

    # Coordenadas de la mejor configuración para el texto
    best_eps = best_configuration['eps']
    best_min_points = best_configuration['minPoints']
    best_evaluacion = best_configuration['evaluacion']
else:
    print("No se encontró una configuración óptima.")

# Crear gráfico 3D con tamaño más grande y mejor calidad
fig = plt.figure(figsize=(12, 8), dpi=150)  # Tamaño 12x8 pulgadas, resolución 150 DPI
ax = fig.add_subplot(111, projection='3d')

# Usar el número de componentes PCA para el color de los puntos
scatter = ax.scatter(eps_values, min_points_values, evaluacion_values,
                     c=pca_components_values, cmap='viridis', marker='o')

# Añadir barra de color para los componentes PCA
cbar = fig.colorbar(scatter)
cbar.set_label('Número de Componentes PCA')

# Etiquetas y título
ax.set_xlabel('EPS', fontsize=12)
ax.set_ylabel('Min Points', fontsize=12)
ax.set_zlabel('Evaluación', fontsize=12)
ax.set_title('Configuraciones de Clustering y Evaluación', fontsize=14)

# Resaltar el mejor punto con un color especial (por ejemplo, rojo)
if best_configuration:
    ax.scatter(best_eps, best_min_points, best_evaluacion, color='red', s=100, label='Mejor configuración', marker='^')

# Añadir leyenda para el mejor punto
ax.legend()

# Mostrar texto de la mejor configuración debajo del gráfico
if best_configuration:
    fig.text(0.1, 0.02,
             f'Mejor Configuración: PCA: {best_configuration["n_components"]}, EPS: {best_eps}, MinPts: {best_min_points}, Evaluación: {best_evaluacion:.4f}',
             fontsize=10, color='blue')

# Mostrar el gráfico
plt.show()
