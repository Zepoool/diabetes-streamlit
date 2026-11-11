"""Predicción de diabetes (Pima Indians) — prototipo académico en Streamlit."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
COLOR_SI, COLOR_NO, COLOR_BASE = "#7A2E3A", "#B08D57", "#8C6A5A"

st.set_page_config(page_title="Predicción de diabetes", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------- Carga (sin reentrenar)
@st.cache_resource
def cargar_pipeline():
    return joblib.load(BASE / "pipeline_diabetes.joblib")


@st.cache_data
def cargar_datos():
    df = pd.read_csv(BASE / "diabetes.csv")
    cols_cero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_cero] = df[cols_cero].replace(0, np.nan)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data
def cargar_metricas():
    with open(BASE / "metricas.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_prueba():
    return pd.read_csv(BASE / "X_test.csv", index_col="indice_original")


pipeline, df, met, prueba = cargar_pipeline(), cargar_datos(), cargar_metricas(), cargar_prueba()
COLUMNAS = met["columnas"]

# ----------------------------------------------------------------- Variables y validación
# etiqueta, unidad, mínimo y máximo clínicamente posibles, valor por defecto, paso, opcional
VARIABLES = {
    "pregnancies": ("Número de embarazos", "", 0, 20, 2, 1, False),
    "glucose": ("Glucosa en plasma (prueba de tolerancia oral)", "mg/dL", 40, 400, 120, 1, False),
    "bloodpressure": ("Presión arterial diastólica", "mm Hg", 20, 150, 70, 1, True),
    "skinthickness": ("Pliegue cutáneo del tríceps", "mm", 5, 100, 25, 1, True),
    "insulin": ("Insulina sérica a las 2 horas", "µU/mL", 10, 900, 80, 1, True),
    "bmi": ("Índice de masa corporal (IMC)", "kg/m²", 12.0, 70.0, 30.0, 0.1, False),
    "diabetespedigreefunction": ("Función de antecedentes familiares de diabetes", "", 0.05, 3.0, 0.40, 0.01, False),
    "age": ("Edad", "años", 21, 100, 35, 1, False),
}


def validar(valores: dict):
    """Devuelve (errores, advertencias). None = dato no registrado (permitido si es opcional)."""
    errores, avisos = [], []
    for col, (nombre, unidad, vmin, vmax, _, _, opcional) in VARIABLES.items():
        v = valores.get(col)
        if v is None:
            if not opcional:
                errores.append(f"**{nombre}** es obligatoria.")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            errores.append(f"**{nombre}**: el valor debe ser numérico.")
            continue
        if not (vmin <= v <= vmax):
            errores.append(f"**{nombre}**: {v} está fuera del rango posible ({vmin}–{vmax} {unidad}).")
            continue
        if col in ("pregnancies", "age") and float(v) != int(v):
            errores.append(f"**{nombre}**: debe ser un número entero.")
            continue
        omin, omax = met["rangos"][col]
        if not (omin <= v <= omax):
            avisos.append(f"**{nombre}** ({v}) está fuera del rango observado en los datos de entrenamiento "
                          f"({omin:g}–{omax:g}); la predicción es menos confiable.")
    return errores, avisos


def predecir(valores: dict):
    fila = pd.DataFrame([{c: (np.nan if valores[c] is None else valores[c]) for c in COLUMNAS}])[COLUMNAS]
    proba = float(pipeline.predict_proba(fila)[0, 1])
    clase = int(pipeline.predict(fila)[0])
    return clase, proba


def estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ----------------------------------------------------------------- Encabezado
st.title("Predicción de diabetes a partir de indicadores clínicos")
st.caption("Prototipo académico · Minería de Datos · Universidad de la Costa (CUC)")
st.warning("Esta aplicación es un **prototipo académico**. No sustituye el criterio ni el diagnóstico "
           "de un profesional de la salud.")

tab1, tab2, tab3, tab4 = st.tabs(["Proyecto", "Exploración de datos", "Predicción", "Modelo y métricas"])

# ----------------------------------------------------------------- 1. Proyecto
with tab1:
    st.subheader("Objetivo")
    st.write("Predecir si una paciente presenta diabetes a partir de indicadores clínicos sencillos "
             "(glucosa, presión arterial, IMC, edad, entre otros) y comparar qué modelo de clasificación "
             "ofrece el mejor equilibrio entre detección de casos positivos e interpretabilidad.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Integrantes**")
        st.markdown("- Fabián López\n- Jonathan Cárdenas\n- Luis Cárdenas")
        st.markdown("**Tipo de problema:** clasificación binaria (aprendizaje supervisado).")
    with c2:
        st.markdown("**Fuente de los datos**")
        st.write("*Pima Indians Diabetes Database* — NIDDK (Smith et al., 1988), disponible en "
                 "[Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). "
                 "768 registros de mujeres de al menos 21 años de origen Pima, con 8 variables "
                 "predictoras y la variable objetivo `outcome` (1 = diabetes, 0 = no diabetes).")
    st.info(f"Modelo final: **{met['modelo']}** (imputación por mediana + escalado + regresión logística). "
            "Se eligió priorizando el recall de la clase diabetes, porque un falso negativo "
            "(no detectar un caso) es más costoso que un falso positivo en un contexto de tamizaje.")

# ----------------------------------------------------------------- 2. EDA
with tab2:
    st.subheader("Visualizaciones del análisis exploratorio")
    g1, g2, g3 = st.columns(3)

    with g1:
        conteo = df["outcome"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.bar(["No diabetes", "Diabetes"], conteo.values, color=[COLOR_NO, COLOR_SI])
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 8, f"{v} ({v / len(df) * 100:.1f}%)", ha="center", fontsize=9)
        ax.set_ylabel("Pacientes"); ax.set_title("Distribución de la variable objetivo", fontsize=10)
        ax.set_ylim(0, conteo.max() * 1.15); estilo(ax)
        st.pyplot(fig); plt.close(fig)
        st.caption("Desbalance moderado: 34,9 % de casos positivos. Por eso no basta con el accuracy.")

    with g2:
        fig, ax = plt.subplots(figsize=(4, 3.4))
        bins = np.linspace(df["glucose"].min(), df["glucose"].max(), 25)
        for val, col, nom in [(0, COLOR_NO, "No diabetes"), (1, COLOR_SI, "Diabetes")]:
            ax.hist(df.loc[df["outcome"] == val, "glucose"].dropna(), bins=bins, density=True,
                    alpha=0.6, color=col, label=nom)
        ax.set_xlabel("Glucosa (mg/dL)"); ax.set_ylabel("Densidad")
        ax.set_title("Glucosa según el resultado", fontsize=10); ax.legend(fontsize=8); estilo(ax)
        st.pyplot(fig); plt.close(fig)
        st.caption("La glucosa es la variable que mejor separa los dos grupos.")

    with g3:
        corr = df.corr()["outcome"].drop("outcome").sort_values()
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.barh(corr.index, corr.values, color=COLOR_SI)
        ax.set_xlabel("Correlación de Pearson con outcome")
        ax.set_title("Asociación con la diabetes", fontsize=10); ax.tick_params(axis="y", labelsize=7); estilo(ax)
        st.pyplot(fig); plt.close(fig)
        st.caption("Asociación, no causalidad. Glucosa (0,50) e IMC (0,31) lideran.")

    faltantes = (df.drop(columns="outcome").isna().mean() * 100).round(1)
    faltantes = faltantes[faltantes > 0]
    st.markdown("**Datos no registrados** (ceros imposibles tratados como faltantes): "
                + ", ".join(f"`{k}` {v} %" for k, v in faltantes.items()) + ".")

# ----------------------------------------------------------------- 3. Predicción
with tab3:
    st.subheader("Ingresar datos de la paciente")
    st.caption("Las variables con casilla “no disponible” pueden dejarse sin registrar: el modelo las "
               "reemplaza por la mediana del entrenamiento.")

    with st.form("formulario"):
        valores, cols = {}, st.columns(2)
        for i, (col, (nombre, unidad, vmin, vmax, defecto, paso, opcional)) in enumerate(VARIABLES.items()):
            with cols[i % 2]:
                etiqueta = f"{nombre} ({unidad})" if unidad else nombre
                es_entero = col in ("pregnancies", "age")
                no_disp = st.checkbox("No disponible", key=f"nd_{col}") if opcional else False
                valor = st.number_input(
                    etiqueta, value=int(defecto) if es_entero else float(defecto),
                    min_value=0 if es_entero else 0.0, max_value=None, step=int(paso) if es_entero else float(paso),
                    key=f"in_{col}", disabled=no_disp,
                    help=f"Rango clínicamente posible: {vmin}–{vmax}" if not no_disp else None)
                valores[col] = None if no_disp else valor
        enviar = st.form_submit_button("Calcular predicción", type="primary")

    if enviar:
        errores, avisos = validar(valores)
        if errores:
            st.error("Corrige los siguientes datos antes de predecir:")
            for e in errores:
                st.markdown(f"- {e}")
        else:
            for a in avisos:
                st.warning(a)
            clase, proba = predecir(valores)
            r1, r2 = st.columns([1, 2])
            with r1:
                if clase == 1:
                    st.error("### Predicción: riesgo de diabetes")
                else:
                    st.success("### Predicción: sin diabetes")
            with r2:
                st.metric("Probabilidad estimada de diabetes", f"{proba * 100:.1f} %")
                st.progress(min(max(proba, 0.0), 1.0))
                st.caption("Umbral de decisión: 50 %. La probabilidad es una estimación del modelo, "
                           "no un diagnóstico.")

    with st.expander("Verificación con un registro conocido del conjunto de prueba"):
        st.write("Selecciona un registro de prueba: la app calcula la predicción y la compara con la que "
                 "produjo el notebook para el mismo registro.")
        idx = st.selectbox("Registro (índice original en el dataset)", prueba.index.tolist())
        fila = prueba.loc[idx]
        st.dataframe(fila[COLUMNAS].to_frame("valor").T, hide_index=True)
        vals = {c: (None if pd.isna(fila[c]) else float(fila[c])) for c in COLUMNAS}
        clase_app, proba_app = predecir(vals)
        v1, v2, v3 = st.columns(3)
        v1.metric("Probabilidad (app)", f"{proba_app * 100:.2f} %")
        v2.metric("Probabilidad (notebook)", f"{fila['proba_notebook'] * 100:.2f} %")
        v3.metric("Resultado real", "Diabetes" if fila["outcome_real"] == 1 else "No diabetes")
        if np.isclose(proba_app, fila["proba_notebook"]) and clase_app == int(fila["pred_notebook"]):
            st.success("La predicción de la app coincide con la del notebook.")
        else:
            st.error("La predicción NO coincide con la del notebook.")

# ----------------------------------------------------------------- 4. Modelo
with tab4:
    comp = met["comparacion"][met["modelo"]]
    t = comp["test"]
    st.subheader(f"Métricas del modelo final: {met['modelo']}")
    st.caption(f"Entrenamiento: {met['n_train']} registros · Prueba: {met['n_test']} registros (estratificado, semilla 42).")

    m = st.columns(5)
    for c, (k, n) in zip(m, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                             ("f1", "F1"), ("roc_auc", "AUC")]):
        c.metric(n, f"{t[k]:.3f}")

    a, b = st.columns(2)
    with a:
        cv_tabla = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            "Media ± desv. (5 pliegues)": [f"{comp['cv'][k][0]:.3f} ± {comp['cv'][k][1]:.3f}"
                                           for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]]})
        st.markdown("**Validación cruzada (solo entrenamiento)**")
        st.dataframe(cv_tabla, hide_index=True)
    with b:
        cm = np.array(comp["confusion"])
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(cm, cmap="Reds")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No diabetes", "Diabetes"]); ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusión (prueba)", fontsize=10)
        st.pyplot(fig); plt.close(fig)

    st.markdown(f"De 54 casos con diabetes en prueba, el modelo detectó {cm[1, 1]} y dejó pasar {cm[1, 0]} "
                f"(falsos negativos); además generó {cm[0, 1]} falsos positivos.")

    st.markdown("**Comparación con los demás modelos del segundo corte (prueba)**")
    filas = [{"Modelo": n, **{k.capitalize() if k != "roc_auc" else "AUC": round(v, 3) for k, v in d["test"].items()}}
             for n, d in met["comparacion"].items()]
    st.dataframe(pd.DataFrame(filas), hide_index=True)

    st.subheader("Limitaciones")
    st.markdown(
        "- Aun el mejor modelo deja sin detectar cerca de 3 de cada 10 casos de diabetes.\n"
        "- El conjunto de prueba es pequeño (154 registros): las métricas son inestables.\n"
        "- `insulin` (48,7 %) y `skinthickness` (29,6 %) tienen muchos datos no registrados; la imputación "
        "por mediana puede distorsionar la relación con el resultado.\n"
        "- Los datos son solo de mujeres Pima de 21 años o más: no se generaliza a otras poblaciones.\n"
        "- No se optimizaron hiperparámetros ni el umbral de decisión.\n"
        "- Los hallazgos son asociaciones, no causas. El modelo no sustituye un diagnóstico médico.")
