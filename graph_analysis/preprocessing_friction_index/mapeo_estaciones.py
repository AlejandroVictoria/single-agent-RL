import pandas as pd
import os

def generar_coordenadas_estaciones(ruta_mapeo, ruta_stops, ruta_salida):
    print("--- INICIANDO CRUCE DE ESTACIONES GTFS ---")
    
    # 1. Verificar que los archivos existan
    if not os.path.exists(ruta_mapeo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_mapeo}")
    if not os.path.exists(ruta_stops):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_stops}")

    # 2. Cargar los archivos CSV y TXT
    # Se especifica dtype=str en los IDs para evitar que pandas elimine ceros a la izquierda
    df_mapeo = pd.read_csv(ruta_mapeo, dtype={'stop_id_gtfs': str})
    df_stops = pd.read_csv(ruta_stops, dtype={'stop_id': str})

    # 3. Hacer el cruce (Inner Join)
    # Relacionamos 'stop_id_gtfs' del primer archivo con 'stop_id' del segundo
    df_cruzado = pd.merge(
        df_mapeo,
        df_stops,
        left_on='stop_id_gtfs',
        right_on='stop_id',
        how='inner'
    )

    # 4. Seleccionar únicamente las columnas solicitadas
    columnas_finales = ['id_sumo', 'stop_id', 'stop_name', 'stop_lat', 'stop_lon']
    df_final = df_cruzado[columnas_finales]

    # 5. Exportar el archivo final
    df_final.to_csv(ruta_salida, index=False, encoding='utf-8')

    print(f"[ÉXITO] Archivo generado: '{ruta_salida}'")
    print(f"Total de estaciones procesadas y cruzadas: {len(df_final)}")
    
    # Mostrar una vista previa en consola para confirmar
    print("\nVista previa de los primeros 3 registros:")
    print(df_final.head(3).to_string(index=False))

# ==========================================
# EJECUCIÓN DEL SCRIPT
# ==========================================
if __name__ == "__main__":
    # Nombres de tus archivos de entrada y salida
    ARCHIVO_MAPEO = 'mapeo_estaciones.csv'
    ARCHIVO_STOPS = 'stops.txt'
    ARCHIVO_SALIDA = 'estaciones_qgis.csv'
    
    generar_coordenadas_estaciones(ARCHIVO_MAPEO, ARCHIVO_STOPS, ARCHIVO_SALIDA)