import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import csv

# Cargar el tokenizer y el modelo de Bertweet
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base")

# Leer el archivo de input (suponemos que no tiene header)
input_file = "data/DataI_MD_POST.csv"  # Reemplaza por el nombre de tu archivo
data = pd.read_csv(input_file, header=None, delimiter='|')

# Extraer las frases de la 5ta columna (índice 4)
frases = data[4].tolist()

# Inicializar una lista para guardar los embeddings
embeddings = []

# Procesar cada frase
for frase in frases:
    # Tokenizar la frase
    inputs = tokenizer(frase, return_tensors="pt", padding=True, truncation=True)

    # Obtener el embedding de la frase
    with torch.no_grad():
        outputs = model(**inputs)

    # Extraer el embedding usando el token [CLS]
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Añadir el embedding a la lista
    embeddings.append(embedding)

# Guardar los embeddings en un archivo CSV
output_file = "data/output_embeddings.csv"  # Reemplaza por el nombre de tu archivo de salida
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Escribir cada embedding como una fila
    for emb in embeddings:
        writer.writerow(emb)

print(f"Embeddings guardados en {output_file}")
