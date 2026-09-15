import os
import pandas as pd
import geopandas as gpd

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
SEMILLAS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]
BASE_DIR = "resultados_E_SARSA"
V_MAX = 50.0

# Lista oficial de edges para la Línea 8 del Trolebús
EDGES_LINEA8 = [
    '220963278#1', '220963278#2', '220963278#3', '220963278#4', '220963278#5', 
    '220963278#6', '220963278#7', '220963278#9', '220963278#10', '220963278#11', 
    '220963278#12', '220963278#13', '220963278#14', '220963278#15', '220963278#16', 
    '220963278#17', '220963278#18', '220963278#19', '1233874083#0', '1233874083#1', 
    '853021443#0', '853021443#1', '853021443#2', '853021443#4', '853021443#5', 
    '94058599#0', '94058599#1', '94058599#2', '94058599#4', '94058599#5', 
    '220965030#1', '220965030#2', '220965030#3', '220965030#5', '220965030#6', 
    '220965030#7', '220965030#8', '220965030#9', '220965030#10', '79684993#0', 
    '94058602#1', '248396929#0', '248396929#1', '248396929#2', '248396929#3', 
    '248396929#4', '248396929#5', '248396929#6', '248396929#7', '248396929#8', 
    '248396929#9', '248396929#10', '168102141', '399543056#4', '399543056#5', 
    '399543056#6', '399543056#7', '399543056#8', '399543056#9', '399543056#10', 
    '399543056#12', '399543056#13', '399543056#14', '399543056#15', '399543056#16', 
    '399543056#17', '869889733#0', '869889733#1', '869889733#2', '869889733#3', 
    '399543057#0', '248396934#0', '248396934#1', '248396934#2', '248396934#4', 
    '248396934#5', '248396934#6', '26740407#1', '248396931#0', '248396931#1', 
    '248396931#2', '248396932#1', '248396932#2', '79684994#1', '79684994#3', 
    '79684994#4', '79684994#5', '79684994#7', '79684994#8', '79684994#9', 
    '79684994#10', '79684994#12', '79684994#13', '79684994#14', '79684994#15', 
    '79684994#17', '79684994#18', '79684994#19', '79684994#20', '79684994#22', 
    '79684994#23', '79684994#24', '248396928#1', '94058606#0', '75039578', 
    '94058595#2', '248396927', '248396926#0', '248396926#1', '248396926#2', 
    '248396926#3', '248396926#4', '248396926#5', '248396926#6', '248396926#7', 
    '248396926#8', '248396926#10', '248396926#11', '248396926#12', '248396926#13', 
    '248396926#14', '248396926#15', '94058594#0', '94058594#1', '94058594#2', 
    '220965031#0', '220965031#1', '220965036#0', '220965036#2', '220965036#3', 
    '220965036#4', '220965036#5', '220965036#6', '220965036#7', '395928862#0', 
    '853364425#0', '853364425#1', '853364425#2', '395928861#0', '395928861#1', 
    '853364423#1', '853364423#2', '853364423#3', '853364423#4', '853364423#5', 
    '853364423#6', '853364423#7', '853364423#8', '853364421#1', '684827311#0', 
    '684827311#1'
]

def procesar_friccion_espacial(episodio, semillas, base_dir, gdf_red_proyectada, v_max_kmh=50.0):
    """
    Convierte la telemetría vehicular a puntos proyectados (UTM 14N), hace map matching
    por cercanía contra la red dada y retorna un DataFrame tabular con la fricción promedio.
    """
    lista_trolebuses = []

    # 1. Cargar CSVs de telemetría por semilla
    for seed in semillas:
        ruta_t = os.path.join(base_dir, f"E_SARSA_seed_{seed}", "trolebuses", f"ep_{episodio}.csv")
        if os.path.exists(ruta_t):
            df_t = pd.read_csv(ruta_t, encoding='latin-1')
            lista_trolebuses.append(df_t)

    if not lista_trolebuses:
        raise FileNotFoundError(f"No se encontraron archivos para el episodio {episodio}.")

    df_trolebuses = pd.concat(lista_trolebuses, ignore_index=True)

    # 2. Calcular fricción instantánea (0 a 1)
    df_trolebuses['friccion_inst'] = (1.0 - (df_trolebuses['velocidad_kmh'] / v_max_kmh)).clip(lower=0)

    # 3. Conversión espacial de telemetría a EPSG:32614 (Métricas en metros)
    gdf_puntos = gpd.GeoDataFrame(
        df_trolebuses,
        geometry=gpd.points_from_xy(df_trolebuses['longitud'], df_trolebuses['latitud']),
        crs="EPSG:4326"
    ).to_crs(32614)

    # 4. Map Matching (Spatial Join por cercanía)
    gdf_unido = gpd.sjoin_nearest(gdf_puntos, gdf_red_proyectada[['id', 'geometry']], how='inner')

    # 5. Agrupar y promediar fricción por segmento (Edge ID)
    stats_friccion = gdf_unido.groupby('id').agg(
        friccion_media=('friccion_inst', 'mean')
    ).reset_index()

    return stats_friccion

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Cargar la red base de SUMO y filtrar la ruta oficial limpia
    gdf_red_base = gpd.read_file('red_zacatenco.geojson')
    
    # Filtrar únicamente los tramos pertenecientes a la Línea 8 sin duplicados
    red_linea8 = gdf_red_base[gdf_red_base['id'].isin(EDGES_LINEA8)].drop_duplicates(subset=['id']).copy()

    # Proyectar la red a UTM Zona 14N para el spatial join en metros
    red_linea8_32614 = red_linea8.to_crs(32614)

    # 2. Obtener los datos tabulares de fricción para ambos episodios
    df_base = procesar_friccion_espacial("000", SEMILLAS, BASE_DIR, red_linea8_32614, v_max_kmh=V_MAX)
    df_base = df_base.rename(columns={'friccion_media': 'friccion_base'})

    df_esarsa = procesar_friccion_espacial("5999", SEMILLAS, BASE_DIR, red_linea8_32614, v_max_kmh=V_MAX)
    df_esarsa = df_esarsa.rename(columns={'friccion_media': 'friccion_esarsa'})

    # 3. Fusión de métricas tabulares por ID de segmento
    df_comparativo = pd.merge(df_base, df_esarsa, on='id', how='outer')

    # 4. Unir métricas sobre la red oficial completa (Garantiza mantener el 100% de geometrías)
    gdf_final = red_linea8.merge(df_comparativo, on='id', how='left')

    # 5. Rellenar con 0.0 los tramos que no registraron paso de unidades
    gdf_final['friccion_base'] = gdf_final['friccion_base'].fillna(0.0)
    gdf_final['friccion_esarsa'] = gdf_final['friccion_esarsa'].fillna(0.0)

    # 6. Re-proyectar a EPSG:4326 y exportar el GeoJSON unificado
    gdf_final = gdf_final.to_crs(4326)
    gdf_final.to_file("analisis_friccion_tramos_L8_filtrado.geojson", driver="GeoJSON")
    
    print("Procesamiento finalizado con éxito.")
    print(f"Total de tramos guardados en GeoJSON: {len(gdf_final)} (coincide con los tramos oficiales).")