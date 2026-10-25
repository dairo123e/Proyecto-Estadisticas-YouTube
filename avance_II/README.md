# Proyecto Estadísticas de YouTube: Avances II

Proyecto académico orientado al análisis estadístico y la aplicación de modelos de aprendizaje automático para clasificar la popularidad de videos de YouTube a partir de métricas de interacción y características agregadas de sus comentarios.

## Integrantes

- Dairo Enrique Contreras Quintana
- Stiven David Álvarez Olmos

## Docente

Paul Alexander Díaz Montaña

## Objetivo

Analizar la información estadística de videos y comentarios de YouTube para identificar las variables relacionadas con su popularidad, comparar diferentes modelos de clasificación y seleccionar el modelo con mejor desempeño.

La popularidad se definió utilizando la mediana de visualizaciones del conjunto de datos:

- **0:** popularidad baja.
- **1:** popularidad alta.

## Conjuntos de datos

El proyecto utiliza dos archivos principales:

- `videos-stats.csv`: contiene información sobre los videos, como título, fecha de publicación, categoría, likes, comentarios y visualizaciones.
- `comments.csv`: contiene comentarios asociados a los videos, likes de los comentarios y sentimiento codificado.

Después de la limpieza e integración se obtuvo un dataset procesado de **1.859 videos y 25 variables**.

## Metodología

El desarrollo del proyecto incluyó las siguientes etapas:

1. Carga y exploración de los datasets.
2. Tratamiento de valores faltantes e inválidos.
3. Eliminación de registros duplicados.
4. Integración de videos y comentarios mediante `Video ID`.
5. Creación de variables agregadas de sentimiento.
6. Definición del nivel de popularidad.
7. Análisis estadístico descriptivo.
8. Correlación de Spearman.
9. Prueba U de Mann-Whitney.
10. Selección de variables predictoras.
11. Entrenamiento y comparación de cinco modelos.
12. Evaluación mediante métricas y validación cruzada.
13. Selección del modelo definitivo.

## Modelos evaluados

Se probaron las siguientes alternativas:

- Modelo base.
- Regresión logística.
- Árbol de decisión.
- Random Forest.
- Gradient Boosting.

## Resultados principales

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.9409 | 0.9505 | 0.9301 | 0.9402 | 0.9880 |
| Regresión logística | 0.9328 | 0.9305 | 0.9355 | 0.9330 | 0.9851 |
| Random Forest | 0.9274 | 0.9441 | 0.9086 | 0.9260 | 0.9860 |
| Árbol de decisión | 0.9167 | 0.9282 | 0.9032 | 0.9155 | 0.9794 |
| Modelo base | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## Modelo seleccionado

El modelo seleccionado fue **Gradient Boosting**, debido a que obtuvo el mejor desempeño general:

- Accuracy: **0.9409**
- Precision: **0.9505**
- Recall: **0.9301**
- F1-score: **0.9402**
- ROC-AUC: **0.9880**

El modelo clasificó correctamente **350 de 372 videos** del conjunto de prueba.

En la validación cruzada de cinco particiones obtuvo un F1-score promedio de **0.9386**, con una desviación estándar de **0.0053**, lo que demuestra un comportamiento estable.

## Variables más importantes

Las variables con mayor influencia en el modelo fueron:

1. Likes del video: **0.8568**
2. Antigüedad del video: **0.0643**
3. Cantidad de comentarios del video: **0.0170**
4. Categoría del video.
5. Proporción de comentarios positivos.
6. Promedio de likes en los comentarios.
7. Longitud del título.
