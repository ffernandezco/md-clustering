import os
import re
import csv

# Ruta de la carpeta que contiene las evaluaciones
folder_path = 'data/sentiments'

# Expresión regular para extraer el PCA, eps y minPts del nombre del archivo
pattern = re.compile(r'(.*)-(.*)-(.*)-(.*)-evaluationSENTIMENTS\.txt')

# Lista para guardar los datos extraídos
data = []

# Iteramos sobre los archivos en la carpeta
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        pca = float(match.group(1))
        eps = float(match.group(2))
        minPts = float(match.group(3))
        metric = str(match.group(4))

        # Ruta completa del archivo de evaluación
        file_path = os.path.join(folder_path, filename)

        # Leemos el contenido del archivo de evaluación
        with open(file_path, 'r') as file:
            content = file.read()

            # Extraemos los valores de Silhouette Index y Davies-Bouldin Index
            silhouette = float(re.search(r'# Silhouette Index:\s*(-?\d+\.\d+)', content).group(1)) if re.search(
                r'# Silhouette Index:\s*(-?\d+\.\d+)', content) else None
            db_index = float(re.search(r'# Davies-Bouldin Index:\s*(\d+\.\d+)', content).group(1)) if re.search(
                r'# Davies-Bouldin Index:\s*(\d+\.\d+)', content) else None

            # Extraemos los ARIs individuales de las variables
            var1 = float(re.search(r'# Class-to-cluster evaluation para la variable 1:\s*(\d+\.\d+)', content).group(
                1)) if re.search(r'# Class-to-cluster evaluation para la variable 1:\s*(\d+\.\d+)', content) else None
            var2 = float(re.search(r'# Class-to-cluster evaluation para la variable 2:\s*(\d+\.\d+)', content).group(
                1)) if re.search(r'# Class-to-cluster evaluation para la variable 2:\s*(\d+\.\d+)', content) else None
            var3 = float(re.search(r'# Class-to-cluster evaluation para la variable 3:\s*(\d+\.\d+)', content).group(
                1)) if re.search(r'# Class-to-cluster evaluation para la variable 3:\s*(\d+\.\d+)', content) else None

        # Añadimos los datos a la lista
        data.append([pca, eps, minPts, metric, silhouette, db_index, var1, var2, var3])

# Guardamos los datos en un archivo CSV
csv_file = 'data/sentiments/resultados_evaluaciones.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Escribimos el encabezado
    writer.writerow(['PCA', 'eps', 'minPts', 'metric', 'Silhouette Index', 'Davies-Bouldin Index',
                     'class-to-cluster var 1', 'class-to-cluster var 2', 'class-to-cluster var 3'])
    # Escribimos los datos
    writer.writerows(data)

print(f'Los datos se han guardado en {csv_file}')
