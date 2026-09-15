import os
import pandas as pd

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================
ALGORITHMS = ["Expected SARSA", "SARSA", "Q-Learning"]

DIR_NAMES = {
    "Expected SARSA": "E_SARSA", 
    "SARSA": "SARSA",
    "Q-Learning": "TD"
}

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]

# Parámetros extraídos de tu función de recompensa
TAU = 1600.0
DELTA = 800.0
LIMITE_INF = TAU - DELTA  # 800.0 m
LIMITE_SUP = TAU + DELTA  # 2400.0 m

# ==============================================================================
# 2. FUNCIÓN DE CÁLCULO CORE
# ==============================================================================
def calcular_metricas_episodio(archivo_parquet, num_episodio):
    """Procesa los parquets de un episodio específico y devuelve métricas puras."""
    resultados_temp = []
    
    for alg in ALGORITHMS:
        total_observaciones = 0
        violaciones_delta = 0
        distancia_minima_global = float('inf')
        distancias_totales = []
        medias_por_semilla = []

        for seed in SEEDS:
            ruta = os.path.join(f"resultados_{DIR_NAMES[alg]}", f"{DIR_NAMES[alg]}_seed_{seed}", archivo_parquet)
            if not os.path.exists(ruta):
                continue
                
            df = pd.read_parquet(ruta)
            df = df[df['episodio'] == num_episodio]
            
            # 1. Extraer distancias físicas válidas (Ignorando dummy <= 0)
            distancias_validas = df.loc[df['dist_lider_float'] > 0, 'dist_lider_float'].dropna()
            
            if not distancias_validas.empty:
                distancias_totales.append(distancias_validas)
                total_observaciones += len(distancias_validas)
                medias_por_semilla.append(distancias_validas.mean())
                
                # 2. Calcular violaciones al umbral delta
                violaciones = ((distancias_validas < LIMITE_INF) | (distancias_validas > LIMITE_SUP)).sum()
                violaciones_delta += violaciones
                
                # 3. Distancia mínima
                min_dist = distancias_validas.min()
                if min_dist < distancia_minima_global:
                    distancia_minima_global = min_dist

        # Cálculos de consolidación matemática
        porcentaje_exito = 0.0
        if total_observaciones > 0:
            porcentaje_exito = ((total_observaciones - violaciones_delta) / total_observaciones) * 100

        if distancias_totales:
            vector_distancias = pd.concat(distancias_totales)
            media_dist = vector_distancias.mean()
            s_medias = pd.Series(medias_por_semilla)
            desviacion_medias = s_medias.std()
            cv_dist = vector_distancias.std() / media_dist if media_dist > 0 else 0.0
        else:
            media_dist, cv_dist = 0.0, 0.0
            distancia_minima_global = 0.0

        resultados_temp.append({
            "Algoritmo": alg,
            "Total Decisiones": total_observaciones,
            "Violaciones a Delta": violaciones_delta,
            "Exito_Num": porcentaje_exito, 
            "Distancia Mín. (m)": distancia_minima_global,
            "Distancia Media (m)": media_dist,
            "Cv": cv_dist,
            "Desviación de Medias": desviacion_medias
        })
        
    return pd.DataFrame(resultados_temp)

# ==============================================================================
# 3. PROCESAMIENTO: LÍNEA BASE (EP 0) Y ESTADO CONVERGIDO (EP 5999)
# ==============================================================================
print("Calculando métricas espaciales y consolidando tabla...\n")

# A) Calcular Episodio 0 y promediar para construir la Línea Base
df_ep0 = calcular_metricas_episodio("monitoreo_acciones_ep_0.parquet", 0)

linea_base = {
    "Algoritmo": "Línea Base (Aleatoria)",
    "Total Decisiones": round(df_ep0["Total Decisiones"].mean(), 1),
    "Violaciones a Delta": round(df_ep0["Violaciones a Delta"].mean(), 1),
    "% Cumplimiento Operativo": f"{df_ep0['Exito_Num'].mean():.2f}%",
    "Distancia Mín. (m)": round(df_ep0["Distancia Mín. (m)"].mean(), 2),
    "Distancia Media (m)": round(df_ep0["Distancia Media (m)"].mean(), 2),
    "Desviación de Medias": round(df_ep0["Desviación de Medias"].mean(), 4),
    "Cv": round(df_ep0["Cv"].mean(), 4)
}

# B) Calcular Episodio 5999 para los agentes entrenados
df_ep5999 = calcular_metricas_episodio("monitoreo_acciones_ep_5999.parquet", 5999)

# Dar formato estético a las filas del Ep 5999
resultados_formateados = []
for _, row in df_ep5999.iterrows():
    resultados_formateados.append({
        "Algoritmo": row["Algoritmo"],
        "Total Decisiones": round(row["Total Decisiones"], 1),
        "Violaciones a Delta": round(row["Violaciones a Delta"], 1),
        "% Cumplimiento Operativo": f"{row['Exito_Num']:.2f}%",
        "Distancia Mín. (m)": round(row["Distancia Mín. (m)"], 2),
        "Distancia Media (m)": round(row["Distancia Media (m)"], 2),
        "Cv": round(row["Cv"], 4),
        "Desviación de Medias": round(row["Desviación de Medias"], 4)
    })

# ==============================================================================
# 4. CONSOLIDACIÓN Y EXPORTACIÓN FINAL
# ==============================================================================
# Unir la Línea Base en la parte superior, seguida de los algoritmos
df_final = pd.DataFrame([linea_base] + resultados_formateados)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(df_final.to_string(index=False))
df_final.to_csv("cumplimiento_delta_completo_dvs.csv", index=False)
print("\nExportación exitosa: 'cumplimiento_delta_completo_dvs.csv'")