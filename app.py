"""
Aprobación de Créditos — app Streamlit
---------------------------------------------------------------------------
Misma fórmula y mismos coeficientes que entrenar_modelo.py (INTERCEPTO y
COEFICIENTES). No se modifican aquí.
"""
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Coeficientes del modelo (idénticos a entrenar_modelo.py)
# ---------------------------------------------------------------------------
INTERCEPTO = 0.286
COEFICIENTES = {
    "Hombre": -0.01,
    "Casados": -0.02,
    "Separados": -0.01,
    "Ingresos": 0.8,               # multiplica a ln(1 + Ingresos)
    "Ratio_Gastos_Ingreso": -0.055,
    "Ratio_Deuda_Ingreso": -0.055,
}


def predict_proba(ingresos, gastos, deuda, hombre=0, casados=0, separados=0):
    """Misma fórmula que predict_proba() en entrenar_modelo.py."""
    ingresos_t = np.log1p(max(ingresos, 0))
    z = (INTERCEPTO
         + COEFICIENTES["Hombre"] * hombre
         + COEFICIENTES["Casados"] * casados
         + COEFICIENTES["Separados"] * separados
         + COEFICIENTES["Ingresos"] * ingresos_t
         + COEFICIENTES["Ratio_Gastos_Ingreso"] * gastos
         + COEFICIENTES["Ratio_Deuda_Ingreso"] * deuda)
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.title("Aprobación de Créditos")
st.caption("Herramienta educativa de análisis crediticio · no constituye asesoría financiera")

genero = st.selectbox("Género", ["Hombre", "Mujer"])
estado_civil = st.selectbox("Estado civil", ["Casado", "Soltero", "Separado"])
ingresos = st.number_input("Ingresos", value=60.0, step=1.0)
gastos = st.number_input("Ratio de gastos (%)", min_value=0.0, max_value=100.0, value=26.0, step=1.0)
deuda = st.number_input("Ratio de deuda (%)", min_value=0.0, max_value=100.0, value=33.0, step=1.0)

if st.button("Predecir"):
    hombre = 1 if genero == "Hombre" else 0
    casados = 1 if estado_civil == "Casado" else 0
    separados = 1 if estado_civil == "Separado" else 0

    prob = predict_proba(ingresos, gastos, deuda, hombre, casados, separados)
    aprobado = prob >= 0.5

    st.metric("Probabilidad de aprobación", f"{prob * 100:.1f}%")
    if aprobado:
        st.success("✓ CRÉDITO APROBADO")
    else:
        st.error("✕ CRÉDITO DENEGADO")
