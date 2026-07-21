"""
Aprobación de Créditos — app Streamlit
---------------------------------------------------------------------------
Misma fórmula y mismos coeficientes que entrenar_modelo.py (INTERCEPTO y
COEFICIENTES). No se modifican aquí. Solo se corrige la presentación visual:
el HTML se "aplana" antes de pasarlo a st.markdown para que Streamlit lo
renderice como diseño y no como texto/código.
"""
import time

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


def contribuciones(ingresos, gastos, deuda, hombre, casados, separados):
    """Mismo desglose que `contributions` en el <script> del HTML original,
    para mostrar el peso de cada factor en el veredicto."""
    ingresos_t = np.log1p(max(ingresos, 0))
    return {
        "Género (Hombre vs. Mujer)": COEFICIENTES["Hombre"] * hombre,
        "Estado civil": (COEFICIENTES["Casados"] * casados
                          + COEFICIENTES["Separados"] * separados),
        "Ingresos": COEFICIENTES["Ingresos"] * ingresos_t,
        "Ratio Gastos/Ingreso": COEFICIENTES["Ratio_Gastos_Ingreso"] * gastos,
        "Ratio Deuda/Ingreso": COEFICIENTES["Ratio_Deuda_Ingreso"] * deuda,
    }


# ---------------------------------------------------------------------------
# Helper de render: evita que Streamlit interprete el HTML como bloque de
# código. Esto pasaba porque las cadenas multilínea quedaban con una línea
# en blanco al inicio y/o con la sangría propia del código Python (los
# `with`/`if` anidados); Markdown interpreta eso como un bloque <pre><code>
# y muestra las etiquetas literalmente en vez de renderizarlas. Aquí se
# quita toda línea en blanco y toda sangría antes de entregarle el HTML a
# st.markdown, así siempre llega "aplanado" y se renderiza como diseño.
# ---------------------------------------------------------------------------
def render_html(content: str) -> None:
    lines = [line.strip() for line in content.strip().splitlines()]
    flat = "".join(line for line in lines if line != "")
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Aprobación de Créditos", page_icon="◆", layout="wide")

# ---------------------------------------------------------------------------
# Estilos (paleta y tipografías tomadas del HTML original)
# Todo el CSS va encerrado en una sola cadena triple-comillada pasada a
# st.markdown(..., unsafe_allow_html=True); así Streamlit lo interpreta
# como un bloque <style> y no como código Python suelto.
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{
  --black-950:#0a0908; --black-900:#121110; --black-800:#1b1917; --black-700:#262320;
  --cream:#efe6d3; --parchment:#b9ad96; --gold:#b8892b; --gold-bright:#d9a63e;
  --gold-pale:#f0d494; --wine:#5c1a2e; --wine-bright:#9c3352; --wine-soft:#c46a83;
  --line: rgba(217,166,62,0.16);
}
#MainMenu, header[data-testid="stHeader"], footer{visibility:hidden;}
.stApp{
  background:
    radial-gradient(ellipse 900px 500px at 12% -10%, rgba(217,166,62,0.09), transparent 60%),
    radial-gradient(ellipse 700px 550px at 105% 15%, rgba(156,51,82,0.14), transparent 55%),
    var(--black-950);
  color:var(--parchment);
  font-family:'Inter',sans-serif;
}
.block-container{max-width:1080px; padding-top:2.5rem; padding-bottom:4rem;}
.masthead{
  display:flex; justify-content:space-between; align-items:flex-end;
  border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:34px;
  gap:20px; flex-wrap:wrap;
}
.brand{display:flex; align-items:center; gap:14px;}
.brand-mark{
  width:42px; height:42px; border-radius:50%;
  border:1.5px solid var(--gold);
  background:radial-gradient(circle at 35% 30%, rgba(217,166,62,0.18), transparent 70%);
  display:flex; align-items:center; justify-content:center;
  font-family:'Fraunces',serif; font-size:20px; color:var(--gold-bright);
  flex-shrink:0;
}
.brand-text h1{
  font-family:'Fraunces',serif; font-weight:500; font-size:27px; margin:0;
  color:var(--cream); letter-spacing:0.2px;
}
.brand-text p{margin:2px 0 0; font-size:12.5px; color:var(--gold-bright); letter-spacing:0.3px;}
.masthead-right{
  font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--parchment);
  opacity:0.7; text-align:right; line-height:1.6; text-transform:uppercase; letter-spacing:0.08em;
}
.st-key-form_panel, .st-key-result_panel{
  background:linear-gradient(180deg, var(--black-800), var(--black-900));
  border:1px solid var(--line);
  border-radius:14px;
  padding:8px 26px 26px;
}
.section-eyebrow{
  font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:0.12em;
  text-transform:uppercase; color:var(--gold-bright); margin:18px 0 4px;
}
.section-title{
  font-family:'Fraunces',serif; font-size:22px; font-weight:500; color:var(--cream);
  margin:0 0 18px;
}
.field-label{ font-size:13px; font-weight:600; color:var(--parchment); margin:14px 0 6px; }
.hint{font-size:11.5px; color:var(--parchment); opacity:0.65; margin:2px 0 0;}
div[data-testid="stPills"] div[role="radiogroup"]{gap:8px;}
div[data-testid="stPills"] button{
  border-radius:9px !important; border:1px solid var(--line) !important;
  background:rgba(255,255,255,0.02) !important; color:var(--parchment) !important;
  opacity:0.85; font-family:'Inter',sans-serif !important; font-weight:500 !important;
  font-size:13.5px !important;
}
div[data-testid="stPills"] button:hover{border-color:var(--gold) !important;}
div[data-testid="stPills"] button[aria-checked="true"],
div[data-testid="stPills"] button[aria-selected="true"]{
  background:linear-gradient(180deg, rgba(217,166,62,0.24), rgba(217,166,62,0.08)) !important;
  border-color:var(--gold) !important; color:var(--gold-pale) !important; opacity:1;
}
div[data-testid="stNumberInput"] input{
  background:rgba(0,0,0,0.3) !important; border:1px solid var(--line) !important;
  border-radius:9px !important; color:var(--cream) !important;
  font-family:'IBM Plex Mono',monospace !important; font-size:15px !important;
}
div[data-testid="stNumberInput"] > div{background:transparent !important; border:none !important;}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]{
  background:var(--gold-bright) !important; border:3px solid var(--black-900) !important;
  box-shadow:0 0 0 1px var(--gold) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div{ background:var(--gold-bright) !important; }
div[data-testid="stSlider"] [data-baseweb="slider"] > div{ background:var(--black-700) !important; }
div[data-testid="stTickBar"]{display:none;}
div[data-testid="stSliderThumbValue"]{
  color:var(--gold-bright) !important; font-family:'IBM Plex Mono',monospace !important;
}
div[data-testid="stButton"] button{
  width:100%; margin-top:10px;
  background:linear-gradient(180deg, var(--gold-bright), var(--gold)) !important;
  color:var(--black-950) !important; border:none !important; border-radius:9px !important;
  padding:12px !important; font-family:'Inter',sans-serif !important; font-weight:700 !important;
  font-size:14.5px !important; letter-spacing:0.02em;
}
div[data-testid="stButton"] button:hover{ box-shadow:0 8px 22px rgba(217,166,62,0.3); }
div[data-testid="stButton"] button p{color:var(--black-950) !important;}
.verdict-empty{
  text-align:center; padding:34px 10px; color:var(--parchment); opacity:0.6;
  font-size:13.5px; line-height:1.6;
}
.verdict-empty svg{margin-bottom:14px; opacity:0.5;}
.gauge-wrap{text-align:center;}
.gauge-num{ font-family:'Fraunces',serif; font-size:56px; font-weight:500; margin:6px 0 0; line-height:1; }
.gauge-label{
  font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:0.12em;
  text-transform:uppercase; color:var(--parchment); opacity:0.65; margin-top:8px;
}
.verdict-tag{
  display:inline-block; margin-top:18px; padding:9px 20px; border-radius:999px;
  font-size:13px; font-weight:700; letter-spacing:0.03em;
}
.verdict-tag.approved{background:rgba(217,166,62,0.16); color:var(--gold-pale); border:1px solid rgba(217,166,62,0.45);}
.verdict-tag.denied{background:rgba(156,51,82,0.18); color:var(--wine-soft); border:1px solid rgba(156,51,82,0.45);}
.warning-banner{
  margin-top:18px; padding:12px 16px; background:rgba(156,51,82,0.16);
  border:1px solid rgba(156,51,82,0.45); border-radius:9px; color:var(--wine-soft);
  font-size:12.5px; line-height:1.6;
}
.warning-banner strong{color:var(--wine-soft);}
.factor-list{margin-top:26px; padding-top:20px; border-top:1px solid var(--line);}
.factor-list h3{
  font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:0.1em;
  text-transform:uppercase; color:var(--parchment); opacity:0.65; margin:0 0 14px; font-weight:500;
}
.factor-row{
  display:flex; justify-content:space-between; align-items:center; font-size:13px;
  padding:8px 0; border-bottom:1px dashed var(--line);
}
.factor-row:last-child{border-bottom:none;}
.factor-name{color:var(--parchment);}
.factor-bar-track{flex:1; margin:0 12px; height:5px; background:var(--black-700); border-radius:3px; position:relative; overflow:hidden;}
.factor-bar-fill{position:absolute; top:0; bottom:0; border-radius:3px;}
.factor-bar-fill.pos{background:var(--gold-bright); left:50%;}
.factor-bar-fill.neg{background:var(--wine-bright); right:50%;}
.factor-mid{position:absolute; left:50%; top:0; bottom:0; width:1px; background:rgba(239,230,211,0.18);}
.footnote{
  text-align:center; margin-top:36px; font-size:11px; color:var(--parchment);
  opacity:0.5; font-family:'IBM Plex Mono',monospace; letter-spacing:0.05em;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
render_html("""
<div class="masthead">
  <div class="brand">
    <div class="brand-mark">A</div>
    <div class="brand-text">
      <h1>Aprobación de Créditos</h1>
      <p>Motor de decisión crediticia · regresión logística</p>
    </div>
  </div>
  <div class="masthead-right">
    Modelo entrenado localmente (statsmodels Logit)<br>
    2,887 solicitudes históricas · variable objetivo: Aprobado
  </div>
</div>
""")

if "resultado" not in st.session_state:
    st.session_state.resultado = None

col_form, col_result = st.columns([1.15, 0.85], gap="medium")

# ---------------------------------------------------------------------------
# Panel 1 — Formulario
# ---------------------------------------------------------------------------
with col_form:
    with st.container(key="form_panel"):
        render_html('<p class="section-eyebrow">Paso 1</p>')
        render_html('<h2 class="section-title">Datos del solicitante</h2>')

        render_html('<p class="field-label">Género</p>')
        genero = st.pills(
            "Género", ["Hombre", "Mujer"], default="Hombre",
            label_visibility="collapsed", key="genero",
        )
        genero = genero or "Hombre"

        render_html('<p class="field-label">Estado civil</p>')
        estado_civil = st.pills(
            "Estado civil", ["Casado", "Soltero", "Separado"], default="Casado",
            label_visibility="collapsed", key="estado_civil",
        )
        estado_civil = estado_civil or "Soltero"

        render_html('<p class="field-label">Ingresos</p>')
        ingresos = st.number_input(
            "Ingresos", value=60.0, step=1.0,
            label_visibility="collapsed", key="ingresos",
        )

        render_html('<p class="field-label">Ratio Gastos / Ingreso (%)</p>')
        gastos = st.slider(
            "Ratio Gastos/Ingreso", min_value=0, max_value=100, value=26,
            label_visibility="collapsed", key="gastos",
        )
        render_html('<p class="hint">Qué porcentaje del ingreso se destina a gastos fijos — de 0% a 100%</p>')

        render_html('<p class="field-label">Ratio Deuda / Ingreso (%)</p>')
        deuda = st.slider(
            "Ratio Deuda/Ingreso", min_value=0, max_value=100, value=33,
            label_visibility="collapsed", key="deuda",
        )
        render_html('<p class="hint">Qué porcentaje del ingreso se destina a pagar deuda existente — de 0% a 100%</p>')

        evaluar = st.button("Evaluar solicitud", use_container_width=True, key="calc_btn")

# ---------------------------------------------------------------------------
# Cálculo (misma lógica que predict_proba / entrenar_modelo.py — sin cambios)
# ---------------------------------------------------------------------------
if evaluar:
    hombre = 1 if genero == "Hombre" else 0
    casados = 1 if estado_civil == "Casado" else 0
    separados = 1 if estado_civil == "Separado" else 0

    with st.spinner("Evaluando solicitud…"):
        time.sleep(0.5)

    prob = predict_proba(ingresos, gastos, deuda, hombre, casados, separados)
    contrib = contribuciones(ingresos, gastos, deuda, hombre, casados, separados)

    st.session_state.resultado = {
        "prob": prob,
        "contrib": contrib,
        "gastos": gastos,
        "deuda": deuda,
    }

# ---------------------------------------------------------------------------
# Panel 2 — Veredicto
# ---------------------------------------------------------------------------
with col_result:
    with st.container(key="result_panel"):
        render_html('<p class="section-eyebrow">Paso 2</p>')
        render_html('<h2 class="section-title">Veredicto del modelo</h2>')

        resultado = st.session_state.resultado

        if resultado is None:
            render_html("""
            <div class="verdict-empty">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.3">
                <circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/>
              </svg>
              <div>Completa el formulario y pulsa "Evaluar solicitud" para ver el resultado.</div>
            </div>
            """)
        else:
            prob = resultado["prob"]
            contrib = resultado["contrib"]
            pct = prob * 100
            aprobado = prob >= 0.5
            color = "var(--gold-pale)" if aprobado else "var(--wine-soft)"

            warning_html = ""
            g, d = resultado["gastos"], resultado["deuda"]
            if (g + d) > 100:
                warning_html = f"""
                <div class="warning-banner">
                  <strong>⚠ Perfil financieramente inviable:</strong>
                  este solicitante destinaría el {g}% de su ingreso a gastos fijos
                  más el {d}% a deuda ({g + d:.0f}% del ingreso en total).
                  Eso deja al solicitante con más compromisos que ingreso disponible,
                  aunque el modelo no siempre lo refleje en la probabilidad de aprobación.
                </div>
                """

            max_abs = max(max(abs(v) for v in contrib.values()), 0.001)
            factor_rows = ""
            for nombre, val in contrib.items():
                width_pct = min(abs(val) / max_abs * 50, 50)
                cls = "pos" if val >= 0 else "neg"
                factor_rows += f"""
                <div class="factor-row">
                  <span class="factor-name">{nombre}</span>
                  <div class="factor-bar-track">
                    <div class="factor-mid"></div>
                    <div class="factor-bar-fill {cls}" style="width:{width_pct}%"></div>
                  </div>
                </div>
                """

            veredicto_txt = "✓ CRÉDITO APROBADO" if aprobado else "✕ CRÉDITO DENEGADO"
            veredicto_cls = "approved" if aprobado else "denied"

            render_html(f"""
            <div class="gauge-wrap">
              <div class="gauge-num" style="color:{color}">{pct:.1f}%</div>
              <div class="gauge-label">probabilidad de aprobación</div>
              <div class="verdict-tag {veredicto_cls}">{veredicto_txt}</div>
            </div>
            {warning_html}
            <div class="factor-list">
              <h3>Peso de cada factor en esta solicitud</h3>
              {factor_rows}
            </div>
            """)

render_html(
    '<div class="footnote">Herramienta educativa de análisis crediticio · '
    'no constituye asesoría financiera</div>'
)
