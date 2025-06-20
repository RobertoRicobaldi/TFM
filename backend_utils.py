import os
import base64
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF

# =====================
# 🔢  CONSTANTES
# =====================

METRICAS = [
    "Pases/90",
    "Pases hacia adelante/90",
    "Precisión pases, %",
    "Precisión pases hacia adelante, %",
    "Pases largos/90",
    "Precisión pases largos, %",
    "Longitud media pases, m",
]

PESOS = {
    "Pases/90": 1.0,
    "Pases hacia adelante/90": 1.5,
    "Precisión pases, %": 2.0,
    "Precisión pases hacia adelante, %": 2.0,
    "Pases largos/90": 1.0,
    "Precisión pases largos, %": 1.5,
    "Longitud media pases, m": 1.0,
}

# =====================
# 🛠️  FUNCIONES
# =====================

def get_flag_img(pais: str, flags_dir: str) -> str:
    for ext in (".png", ".jpg", ".jpeg", ".gif"):
        path = os.path.join(flags_dir, f"{pais}{ext}")
        if os.path.exists(path):
            with open(path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                return f'<img src="data:image/{ext[1:]};base64,{encoded}" width="32">'
    return ""

def calcular_top_global(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df.dropna(subset=METRICAS + ["Posición", "Edad", "País", "Jugador"]).copy()
    for col in METRICAS:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    scaler = MinMaxScaler()
    df_valid[METRICAS] = scaler.fit_transform(df_valid[METRICAS])
    punt = sum(df_valid[m] * PESOS[m] for m in METRICAS)
    df_valid["Puntuación global"] = 10 * punt / punt.max()
    return (
        df_valid[df_valid["Posición"].str.lower() != "portera"]
        .sort_values("Puntuación global", ascending=False)
        .head(10)
        .copy()
    )

def entrenar_modelo(df_in: pd.DataFrame, flags_dir: str) -> pd.DataFrame:
    base = df_in[df_in["Edad"] <= 20].dropna(subset=METRICAS + ["Jugador", "Edad", "Posición", "País"]).copy()
    for col in METRICAS:
        base[col] = pd.to_numeric(base[col], errors="coerce")
    scaler = MinMaxScaler()
    base[METRICAS] = scaler.fit_transform(base[METRICAS])
    punt = sum(base[m] * PESOS[m] for m in METRICAS)
    base["Puntuación global"] = 10 * punt / punt.max()

    X = base[METRICAS]
    y = (base["Puntuación global"] > 7).astype(int)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    base["Proyección"] = model.predict_proba(X)[:, 1] * 100

    base["Bandera"] = base["País"].apply(lambda p: get_flag_img(p, flags_dir))
    def formatear_metricas(row):
        top3 = row[METRICAS].astype(float).nlargest(3)
        return ", ".join([f"{m} ({row[m]:.2f})" for m in top3.index])
    base["Top 3 Métricas"] = base.apply(formatear_metricas, axis=1)

    cols_show = ["Bandera", "Jugador", "Edad", "Posición", "Top 3 Métricas", "Proyección", "País"]
    return base.sort_values("Proyección", ascending=False)[cols_show]

def exportar_pdf_top10(top_df: pd.DataFrame) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Top 10 Jugadoras • Métricas de Pase", ln=True, align="C")
    pdf.ln(4)
    for _, row in top_df.iterrows():
        texto = (
            f"{row['Jugador']} ({row['País']}), Edad {row['Edad']}, "
            f"Posición: {row['Posición']}, Puntuación: {row['Puntuación global']:.2f}"
        )
        pdf.multi_cell(0, 9, txt=texto.encode("latin-1", "replace").decode("latin-1"))
    salida = os.path.join(os.getcwd(), "top10_jugadoras.pdf")
    pdf.output(salida)
    return salida

def exportar_pdf_sub20(df_sub: pd.DataFrame) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Ranking Sub20 – Proyección ML", ln=True, align="C")
    pdf.ln(4)
    for _, row in df_sub.iterrows():
        texto = (
            f"{row['Jugador']} ({row['País']}), {row['Edad']} años, {row['Posición']} – "
            f"Proyección {row['Proyección']:.1f}% – Top métricas: {row['Top 3 Métricas']}"
        )
        pdf.multi_cell(0, 9, txt=texto.encode("latin-1", "replace").decode("latin-1"))
    salida = os.path.join(os.getcwd(), "ranking_sub20.pdf")
    pdf.output(salida)
    return salida
