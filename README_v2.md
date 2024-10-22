# Clustering de textos mediante DBSCAN
**Ander Vicario, Francisco Fernández y Markel Hernández**

Minería de Datos / Grado en Ingeniería Informática de Gestión y Sistemas de Información

_[UPV-EHU](https://www.ehu.eus) / Curso 2024-25_

## Introducción
Partiendo de un conjunto de datos como _DataI_, este proyecto realiza un preprocesamiento de los datos, así como la tokenización, vectorización y reducción de dimensionalidad mediante PCA, seguido de una tarea de clustering utilizando una versión adaptada del algoritmo DBSCAN.

## Librerías necesarias e inicialización del proyecto
Para poder ejecutar todo el código del proyecto, es necesario instalar las siguientes librerías de Python:
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [transformers](https://huggingface.co/docs/transformers/index)
- [torch](https://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [tensorflow](https://www.tensorflow.org/)
- [nltk](https://www.nltk.org/)
- [emoji](https://pypi.org/project/emoji/)
- [tf-keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [pysentimiento](https://github.com/pysentimiento/pysentimiento)

Adicionalmente, será necesario crear el directorio `data` en la raíz del proyecto, que deberá contener el fichero `DataI_MD.csv` que puede obtenerse de la plataforma eGela y que no ha sido incluido con la finalidad de evitar posibles problemas relacionados con la difusión de datos privados. También será necesario incluir un directorio `result`, en el que se irán almacenando los diferentes resultados asociados.

A grandes rasgos, asumiendo que se cuenta con Python 3 instalado en el equipo, los comandos a ejecutar en una terminal para poder ejecutar todo el proyecto serían los siguientes:
```
python3 -m venv env
source env/bin/activate
pip install numpy pandas transformers torch tqdm scikit-learn matplotlib tensorflow nltk emoji pysentimiento tf-keras
mkdir data
mv DataI_MD.csv data
mkdir result
python3 main.py
```

## Estructura del proyecto
El proyecto Python se basa en la siguiente estructura:
* **Preprocesamiento (`preprocess.py`)**: se encarga de filtrar y limpiear los datos del archivo CSV proporcionado, además de dividir los datos en un conjunto para _train_ y otro para _test_.
* **Vectorización (`vectorize.py`)**: haciendo uso del modelo _[AIDA-UPM/BERTuit-base](https://arxiv.org/abs/2204.03465)_ basado en la arquitectura _Transformers_, se realiza una tokenización y una vectorización de los textos.
* **Clustering con DBSCAN (`model_final.py`)**: basándose en las configuraciones que pueden establecerse en `main.py` (minPoints, eps, métrica de distancia y PCA), el modelo utilizará una versión personalizada del algoritmo DBSCAN para agrupar los textos.
* **Análisis de sentimientos (`add_pysentimiento.py`)**: añade también un análisis de emociones basado en _[pysentimiento](https://github.com/pysentimiento/pysentimiento)_. En la documentación se ofrece información adicional sobre la utilidad de añadir los vectores.
* **Clasificación de resultados (`model_brute_paralel.py`)**: clasifica nuevos datos utilizando el modelo y basándose en la fuerza bruta, permitiendo ver y guardar los resultados.

Adicionalmente, puede hacerse uso de modelos alternativos como `model_lsh.py`, sobre el que se ofrece más información en la documentación. Estos modelos ofrecen buenos resultados pero son más lentos a la hora de ser ejecutados.