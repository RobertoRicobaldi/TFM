
# TFM – Análisis de Fútbol Femenino

> **Autor:** Roberto Ricobaldi  
> **Escuela:** Sports Data Campus – Máster Python aplicado al Fútbol

## Objetivo
Construir una aplicación *Streamlit* capaz de:
1. Control de acceso (login)
2. Visualizar **Home** y **Stats** con filtros interactivos
3. Calcular proyección de jugadoras Sub‑20 mediante **Machine Learning**
4. Exportar rankings a PDF con `fpdf`

## Estructura de carpetas

```
Modulo 11/
├── TFM.py                 # Front‑End (Streamlit)
├── backend_utils.py       # Funciones de back‑end
├── acquisition.ipynb      # Notebook de adquisición de datos
├── eda.ipynb              # Notebook EDA + métricas
├── flags/                 # Banderas PNG
├── data_clean.xlsx        # Versión limpia del dataset
└── data_with_score.xlsx   # Dataset con Puntuación global
```

## Requisitos

```
pip install streamlit pandas scikit-learn plotly fpdf seaborn matplotlib nbformat
```

## Ejecución

```bash
streamlit run TFM.py
```

Usuario / contraseña de prueba: **login / login**

## Datos

Archivo Excel original:  
`Fase Organizativa Principales Ligas Europeas.xlsx`  
(Se puede reemplazar por cualquier otra fuente CSV/API indicándolo en `acquisition.ipynb`)

