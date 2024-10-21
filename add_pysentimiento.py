import csv
import re
import nltk
import pandas as pd
import emoji
from pysentimiento import create_analyzer


def add(input_file, output_file):
    nltk.download('stopwords')

    def convert_emojis_to_text(text):
        return emoji.demojize(text, language='es')

    def clean_text(text):
        text = text.lower()  # Convierte a minúsculas
        text = re.sub(r'[^a-zA-Z0-9áéíóúñü\s:]', '',
                      text)  # Elimina símbolos raros, mantiene letras, números y espacios
        return text

    # Leer los datos
    data = pd.read_csv(input_file, header=None, delimiter="|")
    data = data.values[:, 4]

    # Preprocesar los datos: convertir emoticonos a texto
    data = [clean_text(convert_emojis_to_text(text)) for text in data]

    # Analizar sentimientos
    sentiments = [create_analyzer(task="emotion", lang="es").predict(text) for text in data]
    emotion_probabilities = []

    # Definir el orden de las emociones
    emotion_order = ["joy", "sadness", "surprise", "fear", "anger", "disgust", "others"]

    # Extraer las probabilidades y guardarlas en una lista de listas
    for sentiment in sentiments:
        # Crear una lista para las probabilidades de esta instancia, en el orden deseado
        probs = [sentiment.probas[emotion] for emotion in emotion_order]
        emotion_probabilities.append(probs)

    # Guardar resultados en un archivo CSV
    with open(output_file, "w", newline='') as file:
        writer = csv.writer(file)
        header = ["id"] + [f"prob_{emotion}" for emotion in emotion_order]
        writer.writerow(header)  # Escribir el encabezado
        for id in range(len(data)):
            rounded_emotions = [round(prob, 5) for prob in emotion_probabilities[id]]
            writer.writerow([id] + rounded_emotions)
