# Predicción de diabetes con Streamlit — Minería de Datos, tercer corte

Prototipo académico que despliega el modelo del segundo corte (Pima Indians Diabetes Database) en una
aplicación web. **No sustituye el criterio ni el diagnóstico de un profesional de la salud.**

- **Aplicación pública:** https://diabetes-app-gs3ayjt9upvesnvbarq8ki.streamlit.app
- **Integrantes:** Fabián López, Jonathan Cárdenas, Luis Cárdenas
- **Datos:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (NIDDK, Smith et al., 1988) — 768 registros, 8 predictoras.

## Modelo
Pipeline de scikit-learn: imputación por mediana → `StandardScaler` → regresión logística con
`class_weight="balanced"`. Se entrena solo con el 80 % de entrenamiento (estratificado, `random_state=42`),
por lo que las métricas de la app son las del notebook. La app carga `pipeline_diabetes.joblib`; **no reentrena**.

| Métrica (prueba, n=154) | Valor |
|---|---|
| Accuracy | 0,734 |
| Precision | 0,603 |
| Recall (diabetes) | 0,704 |
| F1 | 0,650 |
| AUC | 0,813 |
| Matriz de confusión | VN 75 · FP 25 · FN 16 · VP 38 |

Validación cruzada (5 pliegues, entrenamiento): accuracy 0,762 ± 0,011 · recall 0,701 ± 0,039 · AUC 0,844 ± 0,016.
No se modificó el modelo respecto al segundo corte.

## Contenido del repositorio
| Archivo | Función |
|---|---|
| `app.py` | Aplicación Streamlit |
| `pipeline_diabetes.joblib` | Pipeline completo serializado |
| `diabetes.csv` | Datos originales (EDA de la app) |
| `X_test.csv` | Conjunto de prueba con las probabilidades del notebook (verificación) |
| `metricas.json` | Métricas, matriz de confusión y rangos |
| `entrenar_modelo.py` | Regenera el pipeline y los archivos auxiliares |
| `test_app.py` | Pruebas: caso normal, entrada inválida, datos no disponibles, registro conocido |
| `requirements.txt` | Dependencias con versión fija |
| `Notebook_Proyecto_Diabetes_Corte_2.ipynb` | Notebook |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest test_app.py   # opcional
```
El archivo `.joblib` se guardó con scikit-learn 1.9.1; si se cambia la versión, ejecutar `python entrenar_modelo.py`.
Al desplegar en Streamlit Community Cloud, elegir Python 3.12 en *Advanced settings*.

## Validación de entradas
- Glucosa, IMC, embarazos, antecedentes y edad son obligatorios. Presión, pliegue cutáneo e insulina pueden marcarse como **no disponibles** (el pipeline imputa la mediana del entrenamiento).
- Se rechazan valores no numéricos, fuera del rango clínicamente posible y no enteros en embarazos y edad (los ceros imposibles se tratan como faltantes).
- Si un valor está fuera del rango observado en el entrenamiento, se muestra una advertencia.

## Limitaciones
Detecta 38 de 54 casos de la prueba (16 falsos negativos); prueba pequeña (154) y por tanto métricas inestables; muchos datos no registrados en insulina (48,7 %) y pliegue cutáneo (29,6 %); solo mujeres Pima ≥ 21 años; sin optimización de hiperparámetros ni del umbral; asociaciones, no causalidad.

## Uso de inteligencia artificial (tercer corte)

| Elemento | Información |
|---|---|
| Herramienta | Claude (Anthropic), en chat con ejecución de código |
| Propósito | Revisar el rendimiento del modelo del segundo corte y generar el código de la aplicación, el guardado del pipeline, las pruebas y el README |
| Prompt principal | Redacción formal, conservando su contenido: «Con el enunciado de la Actividad 1 del tercer corte (Integración y despliegue del modelo con Streamlit), el notebook del segundo corte y el archivo diabetes.csv, revisa el rendimiento del modelo seleccionado (accuracy, precision, recall, F1, matriz de confusión y validación cruzada) y verifica si hay fuga de información, desbalance u otros problemas del pipeline. Luego reproduce el pipeline completo (imputación por mediana, escalado y regresión logística con `class_weight="balanced"`), serialízalo con joblib y desarrolla una aplicación en Streamlit que cargue el modelo sin reentrenarlo. La aplicación debe incluir presentación del proyecto, visualizaciones del EDA, formulario con validación de rangos y tipos, predicción con probabilidad, métricas finales, matriz de confusión y limitaciones, junto con `requirements.txt`, `README.md` y pruebas de un caso normal, una entrada inválida y un registro del conjunto de prueba que coincida con el notebook.» |
| Verificación | Se reprodujeron las métricas del notebook con `entrenar_modelo.py` (accuracy 0,734, recall 0,704, matriz 75/25/16/38). La predicción de la app coincide con la del notebook en los 154 registros de prueba. Se ejecutaron pruebas automáticas (`test_app.py`). En la URL pública se probaron un caso normal, una entrada inválida (glucosa 900) y un registro del conjunto de prueba, con coincidencia respecto al notebook. Todos los integrantes revisaron el código. |
| Problemas encontrados | (1) El formulario obligaba a llenar insulina y pliegue cutáneo, variables con 48,7 % y 29,6 % de datos no registrados. (2) El pipeline exige columnas en minúsculas y en orden fijo, y el `.joblib` depende de la versión de scikit-learn. (3) El modelo deja 16 falsos negativos de 54 casos, y no se optimizó el umbral ni los hiperparámetros. (4) La carpeta oculta `.streamlit` no se subió a GitHub al arrastrar los archivos. |
| Ajustes realizados | (1) Se añadió la opción "no disponible" para presión, pliegue cutáneo e insulina. (2) Se guardó el pipeline completo, entrenado solo con el conjunto de entrenamiento, y se fijaron las versiones en `requirements.txt`. (3) Se mantuvo el modelo del segundo corte y se documentó la limitación en la app. (4) La carpeta `.streamlit` se creó manualmente en GitHub. |
