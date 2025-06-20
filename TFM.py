import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import base64
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier

# Importaciones del backend
from backend_utils import (
    get_flag_img,
    entrenar_modelo,
    calcular_top_global,
    exportar_pdf_top10,
    exportar_pdf_sub20,
    METRICAS,
    PESOS,
)

# =============== CONFIGURACIÓN ===============
st.set_page_config(page_title="TFM Fútbol Femenino", layout="wide")

# =============== LOGIN ===============
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    user = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        if user == "login" and password == "login":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Usuario/contraseña incorrectos.")
    st.stop()

# =============== RUTAS Y DATOS ===============
EXCEL_PATH = r"C:\Users\roric\Desktop\Backup 24052025\Master Python\Modulo 11\Fase Organizativa Principales Ligas Europeas.xlsx"
FLAGS_DIR  = r"C:\Users\roric\Desktop\Backup 24052025\Master Python\Modulo 11\flags"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel(EXCEL_PATH)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# =============== CONSTANTES ===============
METRICAS = [
    'Pases/90', 'Pases hacia adelante/90', 'Precisión pases, %',
    'Precisión pases hacia adelante, %', 'Pases largos/90',
    'Precisión pases largos, %', 'Longitud media pases, m'
]

PESOS = {
    'Pases/90': 1.0,
    'Pases hacia adelante/90': 1.5,
    'Precisión pases, %': 2.0,
    'Precisión pases hacia adelante, %': 2.0,
    'Pases largos/90': 1.0,
    'Precisión pases largos, %': 1.5,
    'Longitud media pases, m': 1.0
}

# =============== FUNCIONES AUXILIARES ===============
def get_flag_img(pais: str) -> str:
    for ext in ('.png', '.jpg', '.jpeg', '.gif'):
        path = os.path.join(FLAGS_DIR, f"{pais}{ext}")
        if os.path.exists(path):
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                return f'<img src="data:image/{ext[1:]};base64,{encoded}" width="32">'
    return ''

def top_n_metrics(row: pd.Series, n: int = 3) -> str:
    valores = row[METRICAS].astype(float)
    return ", ".join(valores.nlargest(n).index.str.replace("%", "\u202F%"))

# =============== EXPORTACIÓN PDF ===============
def exportar_pdf_top10():
    top_df = calcular_top_global()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Top 10 Jugadoras - Métricas de Pase", ln=True, align="C")
    pdf.ln(4)

    for _, row in top_df.iterrows():
        texto = (
            f"{row['Jugador']} ({row['País']}), Edad {row['Edad']}, "
            f"Posición: {row['Posición']}, Puntuación: {row['Puntuación global']:.2f}"
        )
        # Reemplazar símbolos conflictivos
        texto = (
            texto.replace("•", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace(" ", " ")  # espacio fino unicode
            .replace(" ", " ")  # espacio unicode raro
            .replace("\u202f", " ")  # espacio fino también
        )
        pdf.multi_cell(0, 9, txt=texto.encode("latin-1", "replace").decode("latin-1"))

    salida = os.path.join(os.getcwd(), "top10_jugadoras.pdf")
    pdf.output(salida)
    return salida


def exportar_pdf_sub20(df_sub: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Ranking Sub20 - Proyección ML", ln=True, align="C")
    pdf.ln(4)

    for _, row in df_sub.iterrows():
        pais = row.get("País", "Desconocido")
        texto = (
            f"{row['Jugador']} ({pais}), {row['Edad']} años, {row['Posición']} - "
            f"Proyección {row['Proyección']:.1f}% - Top métricas: {row['Top 3 Métricas']}"
        )
        texto = (
            texto.replace("•", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace(" ", " ")
            .replace("\u202f", " ")
        )
        pdf.multi_cell(0, 9, txt=texto.encode("latin-1", "replace").decode("latin-1"))

    salida = os.path.join(os.getcwd(), "ranking_sub20.pdf")
    pdf.output(salida)
    return salida


# =============== CÁLCULOS ===============
def calcular_top_global():
    df_valid = df.dropna(subset=METRICAS + ['Posición', 'Edad', 'País', 'Jugador']).copy()
    for col in METRICAS:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    scaler = MinMaxScaler()
    df_valid[METRICAS] = scaler.fit_transform(df_valid[METRICAS])
    puntuacion = sum(df_valid[m] * PESOS[m] for m in METRICAS)
    df_valid['Puntuación global'] = 10 * puntuacion / puntuacion.max()
    return df_valid[df_valid['Posición'].str.lower() != 'portera'].sort_values('Puntuación global', ascending=False).head(10).copy()

def entrenar_modelo(df_in: pd.DataFrame) -> pd.DataFrame:
    base = df_in[df_in["Edad"] <= 20].dropna(subset=METRICAS + ["Jugador", "Edad", "Posición", "País"]).copy()
    for col in METRICAS:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    scaler = MinMaxScaler()
    base[METRICAS] = scaler.fit_transform(base[METRICAS])
    puntuacion = sum(base[m] * PESOS[m] for m in METRICAS)
    base["Puntuación global"] = 10 * puntuacion / puntuacion.max()
    X = base[METRICAS]
    y = (base["Puntuación global"] > 7).astype(int)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    base["Proyección"] = model.predict_proba(X)[:, 1] * 100
    base["Bandera"] = base["País"].apply(get_flag_img)
    def formatear_metricas(row):
        top3 = row[METRICAS].astype(float).nlargest(3)
        metricas_html = []
        for metrica, valor in top3.items():
            nombre = metrica.replace("%", "\u202F%")
            metricas_html.append(f"<span style='color:green'><b>{nombre}</b>: {valor:.2f}</span>")
        return ", ".join(metricas_html)

    base["Top 3 Métricas"] = base.apply(formatear_metricas, axis=1)

    return base.sort_values("Proyección", ascending=False)[["Bandera", "Jugador", "Edad", "Posición", "Top 3 Métricas", "Proyección"]]

# =============== STREAMLIT PÁGINAS ===============
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", ["🏠 Home", "📊 Estadísticas", "📈 Comparativa", "📤 Exportar PDF", "🔮 Proyección ML"])

if page == "🏠 Home":
    st.title("📊 Análisis Organizativo en el Fútbol Femenino")
    st.success("Datos cargados correctamente ✅")
    st.markdown("""
    Esta app forma parte del **TFM** y analiza métricas organizativas de más de 3 000 jugadoras en ligas europeas.

    **Funcionalidades**
    - Filtros interactivos
    - Rankings con puntuación ponderada
    - Banderas 🇪🇸🇫🇷🇩🇪
    - Proyección Sub20 con ML
    - Exportación a PDF
    """)

elif page == "📊 Estadísticas":
    st.markdown("## 📊 Análisis Estadístico de Jugadoras")
    paises = st.multiselect("País", sorted(df["País"].dropna().unique()))
    posiciones = st.multiselect("Posición", sorted(df["Posición"].dropna().astype(str).unique()))
    edad_min, edad_max = st.slider("Edad", int(df["Edad"].min()), int(df["Edad"].max()), (18, 25))
    df_filtrado = df.copy()
    if paises:
        df_filtrado = df_filtrado[df_filtrado["País"].isin(paises)]
    if posiciones:
        df_filtrado = df_filtrado[df_filtrado["Posición"].astype(str).isin(posiciones)]
    df_filtrado = df_filtrado[df_filtrado["Edad"].between(edad_min, edad_max)]
    df_valid = df_filtrado.dropna(subset=METRICAS + ["Posición", "Edad", "País", "Jugador"]).copy()
    for col in METRICAS:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    scaler = MinMaxScaler()
    df_valid[METRICAS] = scaler.fit_transform(df_valid[METRICAS])
    puntuacion = sum(df_valid[m] * PESOS[m] for m in METRICAS)
    df_valid["Puntuación global"] = 10 * puntuacion / puntuacion.max()
    top_df = df_valid[df_valid["Posición"].str.lower() != "portera"].sort_values("Puntuación global", ascending=False).head(10)
    top_df["Bandera"] = top_df["País"].apply(get_flag_img)
    columnas = ["Bandera", "Jugador", "País", "Edad", "Posición", "Puntuación global"] + METRICAS
    st.markdown("### 🏆 Top 10 – Métricas de Pase")
    st.write(top_df[columnas].round(2).to_html(escape=False, index=False), unsafe_allow_html=True)

elif page == "📈 Comparativa":
    st.markdown("## 📈 Comparativa entre Jugadoras")
    jugadores = sorted(df["Jugador"].dropna().unique())
    seleccionadas = st.multiselect("Selecciona jugadoras", jugadores, max_selections=3)
    if len(seleccionadas) >= 2:
        df_valid = df[df["Jugador"].isin(seleccionadas)].dropna(subset=METRICAS)
        for col in METRICAS:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
        scaler = MinMaxScaler()
        df_valid[METRICAS] = scaler.fit_transform(df_valid[METRICAS])
        fig = go.Figure()
        for _, row in df_valid.iterrows():
            fig.add_trace(go.Scatterpolar(r=[row[m] for m in METRICAS], theta=METRICAS, fill='toself', name=row["Jugador"]))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona al menos dos jugadoras.")

elif page == "📤 Exportar PDF":
    st.markdown("## 📤 Exportación del ranking a PDF")
    ruta = exportar_pdf_top10()
    with open(ruta, "rb") as f:
        st.download_button("📥 Descargar Top 10 PDF", data=f, file_name="top10_jugadoras.pdf")

elif page == "🔮 Proyección ML":
    st.markdown("## 🔮 Jugadoras Sub20 con mayor proyección (ML)")
    col1, col2, col3 = st.columns(3)
    with col1:
        paises = st.multiselect("País", sorted(df["País"].dropna().unique()))
    with col2:
        posiciones = st.multiselect("Posición", sorted(df["Posición"].dropna().astype(str).unique()))
    with col3:
        edad_min_ml, edad_max_ml = st.slider("Rango de edad", 15, 20, (17, 20))

    df_sub20 = df[df["Edad"].between(edad_min_ml, edad_max_ml)].copy()

    if paises:
        df_sub20 = df_sub20[df_sub20["País"].isin(paises)]
    if posiciones:
        df_sub20 = df_sub20[df_sub20["Posición"].astype(str).isin(posiciones)]

    df_proy = entrenar_modelo(df_sub20).head(20)
    st.write(df_proy.to_html(escape=False, index=False), unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=df_proy["Proyección"],
        y=df_proy["Jugador"],
        orientation='h',
        text=df_proy["Proyección"].apply(lambda x: f"{x:.1f}%"),
        hovertext=df_proy["Top 3 Métricas"]
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Proyección (%)")
    st.plotly_chart(fig, use_container_width=True)

    ruta_pdf_sub = exportar_pdf_sub20(df_proy)
    with open(ruta_pdf_sub, "rb") as f:
        st.download_button("📥 Descargar Ranking Sub20 PDF", data=f, file_name="ranking_sub20.pdf")


