import gymnasium as gym
import numpy as np
import os
import json


from rl_env.agents import QLearningAgent #Importar los agentes desde rl_env.agents [SARSA, Expected SARSA, QLearning]
from monitoring.log_class import ActionLogger

# ==============================================================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO (Sutton & Barto)
# ==============================================================================
# Semillas para la repetibilidad estadística de la tesis
# SEMILLAS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
SEMILLAS = [101]

# EPISODIOS_POR_SEMILLA = 1000

# Dimensiones del Entorno de Zacatenco
# El entorno tiene un espacio de estados de 7x7x4 (4 canales) y 5 acciones posibles.
STATE_SHAPE = (7, 7, 4)
N_ACTIONS = 5 
TIEMPO_LIMITE = 5300

# Hiperparámetros base
ALPHA = 0.1
MIN_ALPHA = 0.01
# Total de episodios: 6000
# Episodios de exploración/decaimiento: 5000
# Episodios de explotación pura: 1000

ALPHA_DECAY = 0.99954
DECAY_RATE = 0.99908
ALPHA_DICT = {
    "alpha": ALPHA,
    "min": MIN_ALPHA,
    "decay": ALPHA_DECAY
}

GAMMA = 0.99
EPSILON_INICIAL = 1.0
MIN_EPSILON = 0.01
EPSILON_DICT = {
    "epsilon": EPSILON_INICIAL,
    "min": MIN_EPSILON,
    "decay": DECAY_RATE
}
# ==============================================================================
# 2. BUCLE PRINCIPAL DE ENTRENAMIENTO
# ==============================================================================
def ejecutar_experimento(tipo_agente, semilla_global, exp_folder, episodios_por_semilla):
    ruta_guardado = os.path.join("data", f"TD_seed_{semilla_global}")
    archivo_estado = os.path.join(ruta_guardado, "estado.json")

    EPISODIOS_POR_SEMILLA = episodios_por_semilla
    experimento_id = f"{tipo_agente}_seed_{semilla_global}"
    print(f"\nIniciando experimento: {experimento_id}")
    
    env = gym.make(
        "ZacatencoTrolebusEnv-v0",
        tiempo_limite=TIEMPO_LIMITE,
        n_acciones=N_ACTIONS,
        espacio_estados=STATE_SHAPE
        )

    #
    # 1. Instanciamos el agente
    agente = QLearningAgent(
        state_shape=STATE_SHAPE, n_actions=N_ACTIONS, 
        alpha=ALPHA_DICT, gamma=GAMMA, epsilon=EPSILON_DICT
    )

    logger_acciones = ActionLogger()

    historial_recompensas = []
    episodio_inicial = 0

    episodios_clave = {
        0, 
        EPISODIOS_POR_SEMILLA // 4, 
        EPISODIOS_POR_SEMILLA // 2, 
        (3 * EPISODIOS_POR_SEMILLA) // 4, 
        EPISODIOS_POR_SEMILLA - 1
    }

    # Revisar si existe un checkpoint previo
    if os.path.exists(archivo_estado):
        print("Se detectó un checkpoint anterior. Restaurando sesión...")
        
        # 1. Cargar metadatos
        with open(archivo_estado, "r") as f:
            estado = json.load(f)
            episodio_inicial = estado["episodio_guardado"]
            agente.epsilon = estado["epsilon_actual"]
            agente.alpha = estado["alpha_actual"]
            
        # 2. Cargar datos pesados
        agente.q_table = np.load(os.path.join(ruta_guardado, "q_table.npy"))
        historial_recompensas = np.load(os.path.join(ruta_guardado, "recompensas.npy")).tolist()
        
        print(f"  > Reanudando desde el episodio {episodio_inicial} con Epsilon={agente.epsilon:.4f}")

    else:
        print("Iniciando entrenamiento desde cero...")

    # IMPORTANTE: Tu bucle ahora debe usar 'episodio_inicial'
    for ep in range(episodio_inicial, EPISODIOS_POR_SEMILLA):
        opciones = {"experimento_id": experimento_id, "episodio": ep, "e_folder": exp_folder}
        obs, info = env.reset(seed=semilla_global + ep, options=opciones)

        # Activar la bandera sólo para episodios clave
        debe_guardar_csv = ep in episodios_clave
        env.unwrapped.grabar_datos = debe_guardar_csv 
        
        terminated = False
        truncated = False
        recompensa_acumulada = 0
        paso = 0 # Variable auxiliar para ActionLogger
        
        # --- BUCLE DE SIMULACIÓN DE EPISODIO ---
        while not (terminated or truncated):
            accion,tipo_sel,q_previo = agente.elegir_accion(obs)

            # Guardado de acciones tomadas y fotografía de contexto:
            datos_continuos = info["datos_continuos"]
            tiempo_holding = float(accion * 15)

            logger_acciones.registrar_paso(
                ep, paso, datos_continuos, obs, accion, tiempo_holding, tipo_sel, q_previo
            )

            next_obs, reward, terminated, truncated, info = env.step(accion)
            recompensa_acumulada += reward
            
            # Se ejecuta la lógica específica del agente
            q_nuevo = agente.actualizar_paso(tuple(np.array(obs).flatten()), accion, reward, tuple(np.array(next_obs).flatten()), terminated)
            logger_acciones.registrar_recompensa_y_update(reward, q_nuevo)
            obs = next_obs

            # Aumento de variable de paso que contabiliza las acciones tomadas
            paso += 1
        
        # --- FIN DEL EPISODIO ---
        historial_recompensas.append(recompensa_acumulada)
        
        agente.actualizar_fin_episodio()

        print(logger_acciones.obtener_resumen_episodio(ep, recompensa_acumulada, agente.epsilon, agente.alpha))
        
        # Guardar datos sólo si son episodios clave
        if debe_guardar_csv:
            env.unwrapped._volcar_a_csv(exp_folder)
            logger_acciones.exportar_parquet(f"data/TD_seed_{semilla_global}/monitoreo_acciones_ep_{ep}.parquet")
            
        
        # Guardado progresivo cada 25 episodios de la tabla q y las recompensas generadas.
        if (ep + 1) % 25 == 0 or (ep + 1) == EPISODIOS_POR_SEMILLA:
            os.makedirs(ruta_guardado, exist_ok=True)
            
            # 1. Guardar el historial de recompensas acumuladas hasta este ep
            np.save(os.path.join(ruta_guardado, "recompensas.npy"), np.array(historial_recompensas))
            
            # 2. Guardar la matriz Q actual del agente
            np.save(os.path.join(ruta_guardado, "q_table.npy"), agente.q_table)

            estado_entrenamiento = {
                    "episodio_guardado": ep + 1,
                    "epsilon_actual": agente.epsilon, # (Asumiendo que es un atributo de tu agente)
                    "alpha_actual": agente.alpha
                }
            with open(os.path.join(ruta_guardado, "estado.json"), "w") as f:
                json.dump(estado_entrenamiento, f)

            if (ep + 1) % 150 == 0 or (ep + 1) == EPISODIOS_POR_SEMILLA:
                print(f"  > Progreso del episodio {ep + 1} guardado en disco exitosamente.")

    env.close()
    print(f"Experimento {experimento_id} completado.")
    print(f"\nGuardando resultados estadísticos del experimento {experimento_id}...")
    ruta_base = f"data/{experimento_id}"
    
    # 1. ¿Hay algún valor distinto de cero en toda la tabla?
    # Sección para validar actualización de tabla Q
    suma_total = np.sum(agente.q_table)
    max_valor = np.max(agente.q_table)
    min_valor = np.min(agente.q_table)
    
    print(f"[DEBUG]: Suma total de la tabla: {suma_total}")
    print(f"[DEBUG]: Valor máximo en la tabla: {max_valor}")
    print(f"[DEBUG]: Valor mínimo en la tabla: {min_valor}")
    
    # 2. Guardado de valores por episodio
    np.save(f"{ruta_base}/q_table.npy", agente.q_table)
    np.save(f"{ruta_base}/recompensas.npy", np.array(historial_recompensas))
    
    print(f"Archivos generados correctamente en {ruta_base}.")
    

if __name__ == "__main__":
    for semilla in SEMILLAS:
        ejecutar_experimento("TD", semilla, "exp_4_CV_discretizacion_recompensa", 10) 
