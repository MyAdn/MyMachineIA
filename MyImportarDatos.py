import pandas as pd
import mysql.connector

# Función para conectar a MySQL
def conectar_mysql():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="MarYed1487.",  # Cambia esto por tu contraseña de MySQL
        database="industrial_ia"
    )

# Función para cargar datos desde Excel/CSV
def cargar_datos(archivo):
    if archivo.endswith(".xlsx"):
        df = pd.read_excel(archivo)
    elif archivo.endswith(".csv"):
        df = pd.read_csv(archivo)
    else:
        raise ValueError("Formato de archivo no válido. Usa '.xlsx' o '.csv'.")
    return df

# Función para insertar datos en MySQL
def insertar_datos(df):
    connection = conectar_mysql()
    cursor = connection.cursor()
    
    # Crear la consulta SQL dinámicamente
    columnas = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    query = f"INSERT INTO datos_genericos ({columnas}) VALUES ({placeholders})"
    
    # Insertar los datos
    datos = [tuple(row) for row in df.values]
    cursor.executemany(query, datos)
    connection.commit()
    connection.close()
    print(f"{len(datos)} registros insertados correctamente.")

# Importar y cargar datos
if __name__ == "__main__":
    archivo = "datos_simulados.csv"  # Cambia a "datos_simulados.csv" si usas CSV
    datos = cargar_datos(archivo)
    insertar_datos(datos)