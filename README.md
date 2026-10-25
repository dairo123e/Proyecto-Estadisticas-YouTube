# Proyecto estadísticas de YouTube

Proyecto académico orientado al análisis y clasificación de la popularidad de videos de YouTube mediante estadística, procesamiento de lenguaje natural y aprendizaje automático.

## Integrantes

- Dairo Enrique Contreras Quintana
- Stiven David Álvarez Olmos

## Organización del repositorio

### Avance II

Análisis estadístico, preparación de datos, selección de variables y comparación inicial de modelos.

[Ver Avance II](./avance_II/)

### Avance III

Iteración y mejora del modelo, ajuste de hiperparámetros, nuevos algoritmos, TF-IDF y comparación de configuraciones.

[Ver Avance III](./avance_III/)

### Avance IV

Evaluación final frente al diseño inicial, retos, soluciones, lecciones aprendidas, despliegue y monitoreo.

[Ver Avance IV](./avance_IV/)

### Datasets

Datos originales y datos procesados utilizados en el proyecto.

[Ver datasets](./datasets/)

## Modelos finales

### Gradient Boosting ajustado

Modelo principal para clasificar la popularidad observada:

- Accuracy: 0.9462
- Precision: 0.9511
- Recall: 0.9409
- F1-score: 0.9459
- ROC-AUC: 0.9877

### XGBoost con TF-IDF

Modelo complementario para el escenario sin likes ni cantidad Recall: 0.8387
- F1-score: 0.8432
- ROC-AUC: 0.9337

## Tecnologías utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- SciPy
- Scikit-learn
- XGBoost
- LightGBM
- TF-IDF
- Gradio
- GitHub

## Ejecución

Cada avance contiene su propio informe, código y resultados. Para probar el producto final, abre el notebook del Avance IV en Google Colab y ejecuta las celdas en orden.
