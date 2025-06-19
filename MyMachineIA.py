import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import schedule
import time
from threading import Thread
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Logo y título
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Sistema de Monitoreo y Optimización Industrial")
with col2:
    st.image("static/MyLogo/calvek.jpeg", width=150)

# Inicialización del estado
for key, default in {
    'reentrenamiento_thread': None,
    'model': None,
    'campos_seleccionados': [],
    'features': [],
    'tipos_umbrales': {},
    'proximo_reentrenamiento': datetime.now() + timedelta(days=30)
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Conexión a MySQL
def conectar_mysql():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="MarYed1487.",
        database="industrial_ia"
    )

def cargar_datos():
    try:
        conn = conectar_mysql()
        df = pd.read_sql("SELECT * FROM datos_genericos", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

def cargar_configuracion():
    try:
        conn = conectar_mysql()
        cfg = pd.read_sql("SELECT * FROM configuracion", conn)
        conn.close()
        return cfg
    except Exception as e:
        st.error(f"Error al cargar configuración: {e}")
        return pd.DataFrame()

def guardar_configuracion(campos, etiquetas, targets, umbrales, tipos):
    try:
        conn = conectar_mysql()
        cur = conn.cursor()
        cur.execute("DELETE FROM configuracion")
        cur.execute("DELETE FROM umbrales")
        for col in campos:
            cur.execute(
                "INSERT INTO configuracion (columna, etiqueta, es_target) VALUES (%s, %s, %s)",
                (col, etiquetas[col], col in targets))
        for t, val in umbrales.items():
            cur.execute(
                "INSERT INTO umbrales (target, umbral, tipo) VALUES (%s, %s, %s)",
                (t, val, tipos[t]))
        conn.commit()
        conn.close()
        st.success("Configuración guardada exitosamente!")
    except Exception as e:
        st.error(f"Error al guardar configuración: {e}")

def entrenar_modelo(datos, features, targets, etiquetas):
    try:
        if not features or not targets:
            st.error("Debes seleccionar al menos un feature y un target.")
            return None
        X = datos[[col for col, et in etiquetas.items() if et in features]]
        y = datos[[col for col, et in etiquetas.items() if et in targets]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success("Modelo entrenado exitosamente!")
        return model
    except Exception as e:
        st.error(f"Error al entrenar el modelo: {e}")
        return None

def reentrenar_modelo():
    try:
        datos = cargar_datos()
        config = cargar_configuracion()
        targets = config[config['es_target']]['columna'].tolist()
        features = config[~config['es_target']]['columna'].tolist()
        model = entrenar_modelo(datos, features, targets, dict(zip(config['columna'], config['etiqueta'])))
        if model:
            st.session_state['model'] = model
            st.session_state['proximo_reentrenamiento'] = datetime.now() + timedelta(days=30)
    except Exception as e:
        st.error(f"Error durante el reentrenamiento: {e}")

def programar_reentrenamiento():
    schedule.every().month.do(reentrenar_modelo)
    while True:
        schedule.run_pending()
        time.sleep(1)

if st.session_state['reentrenamiento_thread'] is None:
    st.session_state['reentrenamiento_thread'] = Thread(target=programar_reentrenamiento)
    st.session_state['reentrenamiento_thread'].start()

def mostrar_importancia_features(model, features):
    if model:
        imp = model.feature_importances_
        df = pd.DataFrame({'Feature': features, 'Importancia': (imp * 100).round(2)}).sort_values(by='Importancia', ascending=False)
        st.write("Importancia de las características (%):")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df)
        with col2:
            fig, ax = plt.subplots()
            df.plot(kind='bar', x='Feature', y='Importancia', ax=ax, legend=False)
            ax.set_ylabel("Importancia (%)")
            st.pyplot(fig)

def encontrar_configuracion_optima(df, targets, features):
    cfg_opt = {}
    for t in targets:
        if st.session_state['tipos_umbrales'].get(t) == "Por encima":
            opt = df.loc[df[t].idxmax()]
        else:
            opt = df.loc[df[t].idxmin()]
        cfg_opt[t] = {f: opt[f] for f in features}
    return cfg_opt

def mostrar_configuracion_optima(cfg_opt):
    st.write("### Configuración Óptima Inicial")
    for t, vals in cfg_opt.items():
        st.write(f"**Target: {t}**")
        st.write("Valores ideales de los features:")
        st.write(vals)

# UI principal
st.title("Sistema de Monitoreo y Optimización Industrial")
datos = cargar_datos()
if not datos.empty:
    st.write("Datos brutos:")
    st.dataframe(datos.head())

    campos = [col for col in datos.columns if col != 'datetime']
    st.write("Seleccionar campos a utilizar:")
    campos_sel = [col for col in campos if st.checkbox(f"Seleccionar {col}", value=(col in st.session_state['campos_seleccionados']))]
    st.session_state['campos_seleccionados'] = campos_sel

    etiquetas = {col: st.text_input(f"Etiqueta para {col}", value=col) for col in campos_sel}
    targets = st.multiselect("Seleccionar targets:", list(etiquetas.values()))
    st.session_state['targets'] = targets

    umbrales = {}
    tipos = {}
    for t in st.session_state['targets']:
        umbrales[t] = st.number_input(f"Umbral para {t}", value=0.0)
        tipos[t] = st.selectbox(f"Activar Alerta cuando {t} esté:", ["Por encima", "Por debajo"], index=0)
    st.session_state['tipos_umbrales'] = tipos

    if st.button("Guardar configuración"):
        guardar_configuracion(campos_sel, etiquetas, [k for k,v in etiquetas.items() if v in targets], umbrales, tipos)

    if st.button("Entrenar modelo"):
        features = [etiquetas[c] for c in campos_sel if etiquetas[c] not in targets]
        model = entrenar_modelo(datos, features, targets, etiquetas)
        if model:
            st.session_state['model'] = model
            st.session_state['features'] = features

    if st.session_state['model']:
        mostrar_importancia_features(st.session_state['model'], st.session_state['features'])
        targets_real = [c for c, e in etiquetas.items() if e in st.session_state['targets']]
        features_real = [c for c, e in etiquetas.items() if e in st.session_state['features']]
        cfg_opt = encontrar_configuracion_optima(datos, targets_real, features_real)
        mostrar_configuracion_optima(cfg_opt)

        st.write("### Predicciones en tiempo real")
        nuevos = {f: [st.slider(f"Valor para {f}", 0.0, 100.0, 50.0, 1.0)] for f in st.session_state['features']}
        nuevos_df = pd.DataFrame(nuevos)
        preds = st.session_state['model'].predict(nuevos_df)

        if len(targets) == 1:
            val = preds[0]
            t = targets[0]
            st.write(f"{t}: {val:.2f}")
            if t in umbrales:
                if tipos[t] == "Por encima" and val > umbrales[t]:
                    st.error(f"¡Alerta! {t} > umbral: {val:.2f}")
                elif tipos[t] == "Por debajo" and val < umbrales[t]:
                    st.error(f"¡Alerta! {t} < umbral: {val:.2f}")
                else:
                    st.success(f"{t} está en rango óptimo: {val:.2f}")
        else:
            for i, t in enumerate(targets):
                val = preds[0][i]
                st.write(f"{t}: {val:.2f}")
                if t in umbrales:
                    if tipos[t] == "Por encima" and val > umbrales[t]:
                        st.error(f"¡Alerta! {t} > umbral: {val:.2f}")
                    elif tipos[t] == "Por debajo" and val < umbrales[t]:
                        st.error(f"¡Alerta! {t} < umbral: {val:.2f}")
                    else:
                        st.success(f"{t} está en rango óptimo: {val:.2f}")

st.write("---")
st.write("### Reentrenamiento Automático")
restante = st.session_state['proximo_reentrenamiento'] - datetime.now()
st.write(f"Próximo reentrenamiento automático en: {restante.days} días, {restante.seconds // 3600} horas")

if st.button("Reentrenar manualmente"):
    reentrenar_modelo()
