"""Reproduce el pipeline del notebook (Corte 2) y guarda los artefactos que usa la app.

Ejecutar una sola vez:  python entrenar_modelo.py
Genera: pipeline_diabetes.joblib, X_test.csv, metricas.json
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

RANDOM_STATE = 42

# --- Datos: igual que el notebook (ceros imposibles -> NaN, columnas en minúsculas) ---
df = pd.read_csv("diabetes.csv")
cols_cero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_cero] = df[cols_cero].replace(0, np.nan)
df.columns = df.columns.str.strip().str.lower()

X = df.drop(columns="outcome")
y = df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def crear_pipeline(modelo, escalar):
    pasos = [("imputador", SimpleImputer(strategy="median"))]
    if escalar:
        pasos.append(("escalador", StandardScaler()))
    pasos.append(("modelo", modelo))
    return Pipeline(pasos)


modelos = {
    "Regresión logística": crear_pipeline(LogisticRegression(max_iter=1000), True),
    "Regresión logística (balanced)": crear_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"), True),
    "Árbol de decisión": crear_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE), False),
    "Árbol de decisión (depth=4)": crear_pipeline(
        DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE), False),
    "Random Forest": crear_pipeline(RandomForestClassifier(random_state=RANDOM_STATE), False),
}
ELEGIDO = "Regresión logística (balanced)"

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
comparacion = {}
for nombre, pipe in modelos.items():
    r = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    comparacion[nombre] = {
        "test": {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
                 "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
                 "roc_auc": roc_auc_score(y_test, proba)},
        "cv": {m: [float(r[f"test_{m}"].mean()), float(r[f"test_{m}"].std())] for m in scoring},
        "confusion": confusion_matrix(y_test, pred).tolist(),
    }

# --- Pipeline final: SOLO entrenamiento (coincide con las métricas del notebook) ---
pipeline = modelos[ELEGIDO]
joblib.dump(pipeline, "pipeline_diabetes.joblib")

# Conjunto de prueba con la probabilidad que calcula el notebook (para verificar la app)
prueba = X_test.copy()
prueba["outcome_real"] = y_test
prueba["proba_notebook"] = pipeline.predict_proba(X_test)[:, 1]
prueba["pred_notebook"] = pipeline.predict(X_test)
prueba.to_csv("X_test.csv", index_label="indice_original")

# Rangos observados (datos válidos) para validar entradas
rangos = {c: [float(X[c].min()), float(X[c].max())] for c in X.columns}

metricas = {
    "modelo": ELEGIDO,
    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "columnas": list(X.columns),
    "rangos": rangos,
    "comparacion": comparacion,
}
with open("metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print("Pipeline guardado. Métricas en prueba del modelo elegido:")
print(json.dumps(comparacion[ELEGIDO]["test"], indent=2))
print("Matriz de confusión:", comparacion[ELEGIDO]["confusion"])
