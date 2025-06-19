# 🧠 MyMachineIA

Sistema de Monitoreo y Optimización Industrial usando IA.

![Logo](static/MyLogo/calvek.jpeg)

## 📌 Descripción

Aplicación desarrollada en [Streamlit](https://streamlit.io/) que:

- Se conecta a una base de datos MySQL.
- Permite seleccionar variables (`features`) y objetivos (`targets`).
- Entrena un modelo de regresión (Random Forest).
- Muestra importancia de variables.
- Genera alertas si los valores predichos superan umbrales definidos.
- Reentrena automáticamente cada 30 días.

## 📂 Estructura

MyMachineIA/
│
├── MyMachineIA.py # App principal de Streamlit
├── cargar_datos.py # Script para insertar datos desde Excel/CSV a MySQL
├── datos_simulados.csv # Dataset de ejemplo (si lo incluyes)
├── requirements.txt # Dependencias del proyecto
└── static/
└── MyLogo/
└── calvek.jpeg # Logo de la empresa

## 🛠️ Requisitos

- Python 3.9
- MySQL corriendo localmente
- Streamlit y dependencias instaladas

Instala los paquetes con:

```bash
pip install -r requirements.txt

## Uso
1. Lanza la app:

streamlit run MyMachineIA.py

2. Inserta datos a MySQL:

python cargar_datos.py
