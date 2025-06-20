# 📊 TFM – Análisis Organizativo en el Fútbol Femenino

Esta aplicación forma parte del Trabajo de Fin de Máster del programa de Python & Big Data en Sports Data Campus. Analiza el rendimiento organizativo de más de 3 000 jugadoras en ligas femeninas europeas.

---

## 🔗 Accesos

- 🌐 **Aplicación online**: [Abrir Dashboard](https://upttakhrqhhymcvvlw9wzw.streamlit.app/)
- 📂 **Repositorio GitHub**: [Ver código](https://github.com/RobertoRicobaldi/TFM)

---

## 🎯 Objetivo

Desarrollar una app profesional de análisis deportivo que integre:
- Evaluación de métricas organizativas por jugadora
- Predicción de talento Sub20 mediante Machine Learning
- Exportación personalizada en PDF
- Visualización avanzada con filtros y gráficos interactivos

---

## 🧱 Estructura del Proyecto

TFM/
│
├── TFM_online.py ← App principal con interfaz Streamlit
├── backend_utils.py ← Funciones reutilizables y lógica de negocio
├── Fase_Organizativa.xlsx ← Dataset principal con +3000 jugadoras
├── flags/ ← Carpeta de banderas por país
│
├── acquisition.ipynb ← Notebook de carga y validación de datos
├── eda.ipynb ← Análisis exploratorio y generación de métricas
├── requirements.txt ← Requisitos para reproducibilidad
└── README.md ← Este archivo

yaml
Copiar
Editar

---

🧠 Funcionalidades
🔐 Login personalizado

🏠 Página de inicio explicativa

📊 Filtros por país, edad y posición

⚙️ Cálculo de puntuación organizativa ponderada

📈 Radar de comparación de jugadoras

🔮 Ranking Sub20 con predicción ML (Random Forest)

📤 Exportación de Top10 y Sub20 a PDF

🌍 Visualización con banderas y métricas destacadas

📚 Fuentes
Elaboración propia con datos extraídos de plataformas deportivas públicas

Estructura y visualización desarrolladas en Python con Streamlit, Pandas, Plotly, FPDF y Scikit-learn

