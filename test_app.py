"""Pruebas rápidas: caso normal, entrada inválida y registro conocido del conjunto de prueba."""
import numpy as np
from streamlit.testing.v1 import AppTest


def correr(cambios=None, no_disp=()):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    for k, v in (cambios or {}).items():
        at.number_input(key=f"in_{k}").set_value(v)
    for k in no_disp:
        at.checkbox(key=f"nd_{k}").check()
    at.button[0].click().run()
    return at


def test_caso_normal():
    at = correr()
    assert not at.exception
    assert any("Probabilidad estimada" in m.label for m in at.metric)
    assert not at.error


def test_entrada_invalida():
    at = correr({"glucose": 900.0})
    textos = " ".join(m.value for m in at.markdown)
    assert "fuera del rango posible" in textos


def test_datos_no_disponibles():
    at = correr(no_disp=("insulin", "skinthickness"))
    assert any("Probabilidad estimada" in m.label for m in at.metric)


def test_registro_conocido():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception
    assert any("coincide con la del notebook" in s.value and "NO" not in s.value for s in at.success)
