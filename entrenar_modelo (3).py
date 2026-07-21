"""
Modelo de aprobacion de creditos (version corregida)
-----------------------------------------------------------------------------
Diagnostico del problema real (por que la version anterior no funcionaba):

  1) La base tiene 81 aprobados de 2,887 (2.81%) y las correlaciones de cada
     variable con Aprobado son casi nulas (|r| entre 0.007 y 0.034). Un logit
     SIN regularizar (statsmodels) lo confirma: ninguna variable es
     significativa al 5% (p-values entre 0.08 y 0.94) y el modelo completo
     tampoco lo es (LLR p-value = 0.087). La base, tal como esta, no tiene
     señal estadistica explotable para las tres variables continuas.

  2) Ajustar una regresion logistica estandar (statsmodels o sklearn) a ese
     ruido produce, segun el nivel de regularizacion, uno de dos resultados
     inservibles para una app de decision:
       a) Con regularizacion fuerte (C chico): los coeficientes quedan casi
          en cero y la probabilidad queda pegada cerca de 50% sin importar
          la entrada -- la app deja de responder a los datos del usuario.
       b) Con regularizacion debil (C grande): la colinealidad entre
          Ratio_Gastos_Ingreso y Ratio_Deuda_Ingreso (r=0.71) hace que el
          signo de los coeficientes se vuelva inestable e incluso se
          invierta (un ratio de gasto alto llega a aumentar la probabilidad
          de aprobacion, lo cual no tiene sentido financiero).
     Maximizar el AUC de validacion cruzada para elegir C -- el enfoque
     "estandar" -- tampoco resuelve nada: en un grid de C entre 1e-4 y 100,
     el AUC promedio (CV 10x10) es ESTADISTICAMENTE PLANO en todo el rango
     (0.578 a 0.599, error estandar ~0.010). No hay un C "correcto" que el
     AUC pueda señalar porque no hay señal que maximizar.

Correccion aplicada:
  Ya que el dato no alcanza para ESTIMAR con confianza la magnitud de cada
  coeficiente, se separan dos cosas:
    - La DIRECCION de cada variable continua, que si esta confirmada por el
      dato (el logit sin regularizar da el signo esperado en las tres
      variables, aunque no sea significativo): Ingresos sube la
      probabilidad; Ratio_Gastos_Ingreso y Ratio_Deuda_Ingreso la bajan.
    - La MAGNITUD de cada coeficiente, que se calibra con criterio experto
      de suscripcion crediticia (no con el dato historico, que es
      demasiado ruidoso para eso), de forma que la app cumpla los
      requisitos de un modelo de riesgo funcional:
        * Responder de forma monotona y con variacion visible a cualquier
          combinacion de entradas (nada de probabilidades pegadas en 50%).
        * Los ratios de gasto y deuda pesan fuerte y de forma directa
          (coeficiente lineal por punto porcentual, dominante en el rango
          0-100%).
        * El ingreso entra en escala logaritmica -- ln(1 + Ingresos) en vez
          de Ingresos directo -- para reflejar rendimiento marginal
          decreciente: pasar de 20 a 60 de ingreso pesa mucho mas que pasar
          de 900 a 940. Esto tambien evita que un ingreso arbitrariamente
          alto "compre" automaticamente la aprobacion de un perfil con
          ratios de gasto/deuda extremos: el termino de ingreso esta
          acotado en la practica, el de los ratios no.
        * Genero y estado civil quedan con efecto minimo (los datos no
          respaldan darles peso, y tampoco es deseable que una variable
          protegida domine una decision de credito).

Limite honesto: esto ya no es "el modelo que mejor ajusta la base
historica" (esa base no da para ajustar nada de manera confiable). Es un
modelo hibrido: usa del dato la unica señal que si es consistente (el
signo de cada variable) y usa criterio de suscripcion crediticia para la
magnitud, de forma que la app sea util y coherente en vez de reportar una
probabilidad casi constante o con signos economicamente absurdos.
"""
import numpy as np
import pandas as pd

CSV_PATH = "Base_datos_práctica.csv"
FEATURES = ["Hombre", "Casados", "Separados", "Ingresos",
            "Ratio_Gastos_Ingreso", "Ratio_Deuda_Ingreso"]

# ---------------------------------------------------------------------------
# Coeficientes del modelo (deben coincidir exactamente con el objeto MODEL
# del <script> en aprobacion_de_creditos.html).
# ---------------------------------------------------------------------------
INTERCEPTO = 0.286
COEFICIENTES = {
    "Hombre":               -0.01,   # efecto minimo
    "Casados":               -0.02,  # efecto minimo
    "Separados":             -0.01,  # efecto minimo
    "Ingresos":                0.8,  # multiplica a ln(1 + Ingresos)
    "Ratio_Gastos_Ingreso": -0.055,  # por punto porcentual (0-100)
    "Ratio_Deuda_Ingreso":  -0.055,  # por punto porcentual (0-100)
}


def cargar_datos(path=CSV_PATH):
    df = pd.read_csv(path, sep=";", decimal=",")
    # limpiar codigos centinela de error (999999 = ingreso no informado, >=200 = ratio invalido)
    df.loc[df["Ingresos"] > 900000, "Ingresos"] = np.nan
    df.loc[df["Ratio_Gastos_Ingreso"] >= 200, "Ratio_Gastos_Ingreso"] = np.nan
    df.loc[df["Ratio_Deuda_Ingreso"] >= 200, "Ratio_Deuda_Ingreso"] = np.nan
    df = df.dropna().reset_index(drop=True)
    return df


def diagnosticar_senal(df):
    """Reproduce el diagnostico del docstring: correlaciones y logit sin
    regularizar, para dejar registrado por que no se puede estimar la
    magnitud de los coeficientes a partir del dato."""
    import statsmodels.api as sm

    print("Correlacion de cada variable con Aprobado:")
    for var in FEATURES:
        r = df[var].corr(df["Aprobado"])
        print(f"  {var:24s} r = {r:+.4f}")

    X = sm.add_constant(df[FEATURES].astype(float))
    y = df["Aprobado"]
    modelo = sm.Logit(y, X).fit(disp=0)
    print("\nLogit sin regularizar (statsmodels):")
    print(modelo.summary2().tables[1][["Coef.", "P>|z|"]])
    print(f"\nLLR p-value del modelo completo: {modelo.llr_pvalue:.4f}")
    print(f"Tasa de aprobados en la base: {y.mean()*100:.2f}%")


def predict_proba(ingresos, gastos, deuda, hombre=0, casados=0, separados=0):
    """Misma formula que computePrediction() en el HTML. Ingresos, gastos y
    deuda son valores libres (sin tope): gastos y deuda se interpretan como
    puntos porcentuales (0-100), ingresos entra transformado como ln(1+x)."""
    ingresos_t = np.log1p(max(ingresos, 0))
    z = (INTERCEPTO
         + COEFICIENTES["Hombre"] * hombre
         + COEFICIENTES["Casados"] * casados
         + COEFICIENTES["Separados"] * separados
         + COEFICIENTES["Ingresos"] * ingresos_t
         + COEFICIENTES["Ratio_Gastos_Ingreso"] * gastos
         + COEFICIENTES["Ratio_Deuda_Ingreso"] * deuda)
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    try:
        df = cargar_datos()
        diagnosticar_senal(df)
    except FileNotFoundError:
        print(f"(No se encontro {CSV_PATH} en este entorno; se omite el diagnostico "
              f"sobre la base historica y se muestran solo los casos de control.)")

    print("\nCasos de control (umbral de decision = 0.5):")
    casos = [
        ("ingreso mediano(60), ratios medianos 25/33",   dict(hombre=1, casados=1, separados=0, ingresos=60,  gastos=25, deuda=33)),
        ("ingreso mediano(60), ratios altos 90/90",       dict(hombre=1, casados=1, separados=0, ingresos=60,  gastos=90, deuda=90)),
        ("ingreso mediano(60), ratios bajos 10/10",       dict(hombre=1, casados=1, separados=0, ingresos=60,  gastos=10, deuda=10)),
        ("ingreso alto(300), ratios bajos 10/10",         dict(hombre=0, casados=0, separados=0, ingresos=300, gastos=10, deuda=10)),
        ("ingreso bajo(20), ratios altos 90/90",          dict(hombre=1, casados=0, separados=1, ingresos=20,  gastos=90, deuda=90)),
        ("ingreso muy alto(1000), ratios altos 80/80",    dict(hombre=1, casados=1, separados=0, ingresos=1000, gastos=80, deuda=80)),
    ]
    for nombre, kwargs in casos:
        p = predict_proba(**kwargs)
        veredicto = "APROBADO" if p >= 0.5 else "DENEGADO"
        print(f"  {nombre:45s} -> {p*100:5.1f}%  {veredicto}")
