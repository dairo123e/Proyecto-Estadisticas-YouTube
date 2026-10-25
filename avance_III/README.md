# Proyecto estadísticas de YouTube: Avance III

Esta carpeta contiene la tercera etapa del proyecto, enfocada en la iteración, mejora y comparación de modelos de clasificación de popularidad.

## Integrantes

- Dairo Enrique Contreras Quintana
- Stiven David Álvarez Olmos

## Iteraciones realizadas

1. Réplica del modelo desarrollado en el Avance II.
2. Ajuste de hiperparámetros y evaluación de nuevos algoritmos.
3. Ingeniería de variables y procesamiento textual mediante TF-IDF y SVD.
4. Ensamble de modelos mediante Stacking.
5. Evaluación de un escenario sin likes ni cantidad de comentarios del video.

## Modelo principal seleccionado

Se seleccionó Gradient Boosting ajustado:

- Accuracy: 0.9462
- Precision: 0.9511
- Recall: 0.9409
- F1-score: 0.9459
- ROC-AUC: 0.9877
- F1 promedio de validación cruzada: 0.9400

## Modelo complementario

Para el escenario sin likes ni cantidad de comentarios del video se utilizó XGBoost con características textuales.

## Archivos principales
- `iteraciones_avance_III.py`
- `resultados_iteraciones.csv`
