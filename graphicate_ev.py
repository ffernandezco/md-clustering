import os
import csv
import re

# Directorio con los archivos evaluation.txt
directory = 'data/result/'  # Cambia esto a la ruta de tu carpeta

# Archivo de salida CSV
output_csv = 'data/resultados.csv'

# Expresiones regulares para extraer datos
config_re = re.compile(r"eps=([\d\.]+),\sminPts=(\d+)")
chi_p_re = re.compile(r"Chi2_(\d)=(\d+\.\d+),\sp_(\d)=(\d+\.\d+)")
avg_re = re.compile(r"Avg_Chi2=(\d+\.\d+),\sAvg_p=(\d+\.\d+)")

# Cabecera del CSV
csv_header = ['eps', 'minPts', 'Chi2_1', 'p_1', 'Chi2_2', 'p_2', 'Chi2_3', 'p_3', 'Avg_Chi2', 'Avg_p']

# Abrir archivo CSV para escritura
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)

    # Recorrer archivos en el directorio
    for filename in os.listdir(directory):
        if filename.endswith('evaluation.txt'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = file.read()

                # Extraer configuración eps y minPts
                config_match = config_re.search(data)
                if config_match:
                    eps, minPts = config_match.groups()

                # Extraer valores Chi2 y p para cada variable
                chi_p_matches = chi_p_re.findall(data)
                chi2_vals = []
                p_vals = []
                for match in chi_p_matches:
                    chi2_vals.append(match[1])
                    p_vals.append(match[3])

                # Extraer evaluación general
                avg_match = avg_re.search(data)
                if avg_match:
                    avg_chi2, avg_p = avg_match.groups()

                # Escribir datos en el CSV
                row = [eps, minPts] + chi2_vals + p_vals + [avg_chi2, avg_p]
                writer.writerow(row)

print(f"Los datos han sido exportados a {output_csv}.")