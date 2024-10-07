# CSV Processor

Este proyecto contiene un script en Python que procesa un archivo CSV para estructurarlo, realizar algunas modificaciones y luego escribir los resultados en un nuevo archivo CSV.

## Estructura del Proyecto

### Clases

- **Instance**: Representa una fila del archivo CSV procesado. Tiene cuatro atributos principales:
  - `pcm`
  - `rs`
  - `pu`
  - `txt`

  Además, tiene métodos `get` y `set` para modificar o acceder a estos atributos.

### Funciones

- **parse_int_or_zero(value)**: Convierte un valor a entero. Si no puede ser convertido, devuelve 0.
  
- **preprocess(input_file, output_file)**: Función principal que procesa un archivo CSV de entrada, realiza modificaciones según la lógica del script y guarda el resultado en un archivo CSV de salida. Realiza los siguientes pasos:
  - Lee el archivo CSV de entrada.
  - Divide las filas en exactamente 5 columnas.
  - Elimina la primera fila (cabecera).
  - Crea objetos `Instance` para cada fila.
  - Elimina instancias donde el valor `ui` sea "0".
  - Aplica reglas para modificar los valores `pcm` y `pu`.
  - Escribe los datos procesados en un archivo CSV de salida.

## Uso

