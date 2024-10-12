import preprocess
import vectorize
import model
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
all_points = model_final.read_csv("data/VECTOR_BERTuit.csv")
configuration = [14, 20]
clusters = model_final.train(all_points, configuration[0], configuration[1])
model_final.plot_clusters(all_points, clusters, configuration)
model_final.save_cluster_vectors_to_csv(all_points, clusters, max_per_cluster=5,
                                        output_csv_path="data/" + str(configuration[0]) + "," + str(
                                            configuration[1]) + "cluster_vectors.csv")
model_final.save_cluster_texts_to_csv("data/" + str(configuration[0]) + "," + str(configuration[1]) +
                                      "cluster_vectors.csv", 'data/DataI_MD_POST.csv', "data/" +
                                      str(configuration[0]) + "," + str(configuration[1]) + "cluster_texts.csv")
