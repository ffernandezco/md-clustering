import preprocess
import vectorize
import model

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
model.train("data/VECTOR_BERTuit.csv")
