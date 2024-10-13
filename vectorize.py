import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import csv
from tqdm import tqdm


def vectorize(input_file, output_file, enable_cuda=False, model="AIDA-UPM/BERTuit-base", from_tf=True, c=4,
              batch_size=64):
    # CUDA se activa en el caso de que sea compatible y se active la opción
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModel.from_pretrained(model, from_tf=from_tf).to(device)
    data = pd.read_csv(input_file, header=None, delimiter='|')

    # Inicializaciones
    frases = data[c].tolist()
    num_batches = len(frases) // batch_size + (1 if len(frases) % batch_size != 0 else 0)
    embeddings = []

    # Procesar las frases por batches con tqdm
    for i in tqdm(range(num_batches), desc="Procesando batches", unit="batch"):
        batch_frases = frases[i * batch_size: (i + 1) * batch_size]
        inputs = tokenizer(batch_frases, return_tensors="pt", padding=True, truncation=True).to(device)

        # Obtener el embedding de las frases en el batch
        with torch.no_grad():
            outputs = model(**inputs)

        # Extraer los embeddings usando el token [CLS] y añadir los embeddings del batch a la lista
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)

    # Guardar los embeddings en un archivo CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for emb in embeddings:
            writer.writerow(emb)

    print(f"Embeddings guardados en {output_file} correctamente.")
