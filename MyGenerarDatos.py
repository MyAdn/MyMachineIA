import pandas as pd
from datetime import datetime, timedelta
import random

# Función para generar datos simulados
def generar_datos_simulados(num_registros=1000):
    datos = []
    for i in range(num_registros):
        # Fecha y hora aleatoria en los últimos 30 días
        fecha = datetime.now() - timedelta(days=random.randint(0, 30))
        
        # Generar 50 valores numéricos aleatorios
        valores_numericos = [random.uniform(0, 100) for _ in range(50)]
        
        # Generar 10 valores alfanuméricos aleatorios
        valores_alfanumericos = [f"Texto_{random.randint(1, 100)}" for _ in range(10)]
        
        # Combinar todo en un registro
        registro = [fecha] + valores_numericos + valores_alfanumericos
        datos.append(registro)
    
    # Crear un DataFrame con los datos
    columnas = ["datetime"] + [f"dato{i}" for i in range(1, 51)] + [f"texto{i}" for i in range(1, 11)]
    df = pd.DataFrame(datos, columns=columnas)
    
    return df

# Guardar los datos en un archivo Excel o CSV
def guardar_datos(df, formato="excel"):
    if formato == "excel":
        df.to_excel("datos_simulados.xlsx", index=False)
        print("Datos guardados en 'datos_simulados.xlsx'")
    elif formato == "csv":
        df.to_csv("datos_simulados.csv", index=False)
        print("Datos guardados en 'datos_simulados.csv'")
    else:
        print("Formato no válido. Usa 'excel' o 'csv'.")

# Generar y guardar datos
if __name__ == "__main__":
    datos_simulados = generar_datos_simulados()
    guardar_datos(datos_simulados, formato="csv")  # Cambia a "csv" o "excel" si prefieres CSV