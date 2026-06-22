# Sudoku DL Solver

Proyecto de Deep Learning para detectar, leer y resolver un Sudoku a partir de una imagen.

## Objetivo

Construir una aplicación capaz de recibir una imagen de Sudoku en formato `PNG`, `JPG` o `JPEG`, detectar el tablero, identificar los números presentes y resolver el Sudoku.

El proyecto incluye una etapa de revisión manual ya que el reconocimiento de dígitos desde imagen puede cometer errores puntuales.

## *Demo* con *Streamlit*

Desde la carpeta principal del proyecto (en terminal):

Ejecutar la app:

```powershell
streamlit run app.py
```

También se puede abrir en el navegador:

```text
http://localhost:8501
```

### Uso de la app

1. Elegir la fuente de imagen en la barra lateral.
2. Subir una imagen `PNG/JPG/JPEG` o usar la imagen de ejemplo.
3. Pulsar **Procesar y resolver**.
4. Revisar la matriz detectada.
5. Corregir manualmente cualquier dígito (si fuera necesario).
6. Pulsar **Resolver Sudoku**.
7. Descargar la solución en CSV si se desea.

## Estructura del proyecto:

### Notebooks

#### `01_yolo_detection.ipynb`

Desarrolla la parte de detección del tablero:

- carga del modelo YOLO,
- detección del tablero,
- recorte de la región detectada,
- redimensionar el tablero,
- división en 81 celdas,
- limpieza de bordes,
- conversion de celdas a formato similar a *MNIST*.

### `02_digit_recognition.ipynb`

Entrena y evalúa una CNN para reconocimiento de dígitos:

- carga del dataset *MNIST*,
- normalización de imágenes,
- entrenamiento de CNN multiclase,
- evaluación del modelo,
- guardado del modelo `digit_cnn.keras`,
- predicción de dígitos en las celdas reales del Sudoku,
- construcción de matriz `9x9`.

### `03_pipeline_demo.ipynb`

Valida la parte de resolución:

- carga de la matriz detectada/corregida,
- validación de formato,
- validación de reglas iniciales,
- resolución con *backtracking*,
- validación de la solución final,
- guardado de solución en `.npy` y `.csv`.

## Módulos principales

### `src/detection.py`

Contiene funciones para:

- cargar el modelo YOLO,
- detectar el tablero,
- seleccionar la caja con mayor confianza,
- recortar el tablero detectado.

### `src/preprocessing.py`

Contiene funciones para:

- redimensionar el tablero,
- dividirlo en 81 celdas,
- limpiar bordes,
- detectar celdas vacías,
- preparar dígitos en formato `28x28` para la CNN.

### `src/digit_recognition.py`

Contiene funciones para:

- cargar el modelo CNN,
- predecir dígitos individuales,
- construir la matriz predicha del Sudoku.

### `src/solver.py`

Contiene funciones para:

- validar la matriz inicial,
- comprobar reglas de filas, columnas y bloques,
- resolver el Sudoku con *backtracking*,
- validar la solución completa.

### `src/pipeline.py`

Conecta las piezas principales del proyecto en un flujo reutilizable.

## Modelos

El proyecto usa dos modelos:

- `best.pt`: modelo YOLO para detectar el tablero del Sudoku.
- `digit_cnn.keras`: CNN entrenada con *MNIST* para reconocer dígitos.

La resolución final se realiza con un algoritmo clásico de *backtracking*, ya que garantiza una solución valida si la matriz inicial no tiene conflictos.

## Resultados

En la imagen de prueba incluida en `data/img.png`, el sistema detecta correctamente el tablero y reconoce la mayoría de los dígitos. Se observaron errores puntuales en dígitos visualmente similares, por lo que la aplicación incluye una <u>fase de revisión manual</u> antes de resolver.

Esta decisión hace que el sistema sea más robusto para una *demo* real: el modelo automatiza la mayor parte del proceso, y el usuario conserva control sobre la matriz antes del solver.

## Dependencias principales

- Streamlit
- Ultralytics YOLO
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

## Limitaciones

- La imagen debe contener un Sudoku claro y detectable.
- La detección puede fallar si el tablero esta muy inclinado, cortado o con baja calidad.
- El reconocimiento de dígitos puede confundir nu¿úmeros visualmente parecidos.
- La app incluye correccion manual para evitar que esos errores bloqueen la resolución.