# Proyecto estadísticas de YouTube: Avance IV

Esta carpeta contiene la cuarta etapa y el cierre del proyecto. El avance compara los resultados finales con el diseño inicial, documenta los retos y soluciones, presenta las lecciones aprendidas y evidencia el trabajo colaborativo.

## Integrantes

- Dairo Enrique Contreras Quintana
- Stiven David Álvarez Olmos

## Producto final

La solución contiene dos modelos:

### Modelo A: Gradient Boosting ajustado

Modelo principal para clasificar la popularidad observada cuando están disponibles las métricas de interacción.

- Accuracy: 0.9462
- Precision: 0.9511
- Recall: 0.9409
- F1-score: 0.9459
- ROC-AUC: 0.9877
- F1 promedio de validación cruzada: 0.9400

### Modelo B: XGBoost con TF-IDF

Modelo complementario para realizar predicciones cuando no se conocen los likes ni la cantidad de comentarios del video.

- Accuracy: 0.8441
- Precision: 0.8478
- Recall: 0.8387
- F1-score: 0.8432
- ROC-AUC: 0.9337

## Funcionalidades

- Interfaz de predicción desarrollada con Gradio.
- Selección del modelo según la información disponible.
- Registro y monitoreo de predicciones.
- Evaluación de deriva mediante PSI.
- Modelos serializados con sus metadatos.
- Ejecución reproducible en Google Colab.

## Archivos principales

- `Proyecto_Estadisticas_YouTube_Avances_IV.ipynb`
- `resultados_finales.csv`
- `errores_por_categoria.csv`
- `monitoreo_predicciones.csv`
- `monitoreo_psi.csv`
- `requirements.txt`
- `modelos/`


