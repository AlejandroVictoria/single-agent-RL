import multiprocessing
import os
import time
import sys
import psutil

# IMPORTANTE: Importamos tu función principal de entrenamiento
from agent_training import ejecutar_experimento

# =====================================================================
# CONFIGURACIÓN DE AFINIDAD PARA EL i7-13620H
# =====================================================================
# Mapeamos las 6 semillas a los 6 P-cores físicos (2 hilos lógicos por core)
# Los hilos 12 al 15 (E-cores) quedan libres para el monitoreo y el OS.
AFINIDAD_PCORES = [
    [0, 1],   # P-core 0 -> Semilla 1
    [2, 3],   # P-core 1 -> Semilla 2
    [4, 5],   # P-core 2 -> Semilla 3
    [6, 7],   # P-core 3 -> Semilla 4
    [8, 9],   # P-core 4 -> Semilla 5
    [10, 11]  # P-core 5 -> Semilla 6
]

def worker_entrenamiento(config):
    """
    Esta función es ejecutada por un worker independiente dentro del Pool.
    Al estar en un proceso aislado, libsumo funcionará sin colisiones.
    """
    seed = config["seed"]
    episodios = config["episodios"]
    tipo_agente = config["tipo_agente"]
    worker_id = config["worker_id"] # ID del 0 al 5 para asignar CPU
    
    # 0. Restringir hilos internos y asignar afinidad de CPU
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(AFINIDAD_PCORES[worker_id])
    except Exception as e:
        # No detenemos el proceso si falla psutil, solo lo advertimos en el log
        pass
    
    # 1. Crear carpeta para los logs si no existe
    os.makedirs("output_logs", exist_ok=True)
    nombre_log = f"output_logs/entrenamiento_{tipo_agente}_seed_{seed}.log"
    
    # Redirección en modo no-buferizado (escribiendo directo al disco)
    log_file = open(nombre_log, "w", encoding="utf-8", buffering=1)
    
    # Guardamos el stdout/stderr original para restaurarlos al final
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    
    print(f"==================================================")
    print(f" [INICIO] PID: {os.getpid()} | Semilla: {seed} | Agente: {tipo_agente}")
    print(f" Hilos CPU asignados: {AFINIDAD_PCORES[worker_id]}")
    print(f" Tiempo de inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================", flush=True)
    
    # 2. Nombre del experimento/carpeta único para esta semilla
    # Ejemplo: exp_LQR_seed_101
    nombre_experimento = f"exp_{tipo_agente}_seed_{seed}"
    
    try:
        # ========================================================
        # 3. LLAMADA A LA FUNCIÓN DE ENTRENAMIENTO
        # Asegúrate de que ejecutar_experimento en tu agent_training.py
        # acepte estos parámetros o adáptalos aquí.
        # ========================================================
        ejecutar_experimento(
            tipo_agente=tipo_agente,
            semilla_global=seed,
            exp_folder="exp_4_CV_discretizacion_recompensa",
            episodios_por_semilla=episodios
        )
        
        print(f"\n [ÉXITO] Entrenamiento de semilla {seed} completado.", flush=True)
        return {"seed": seed, "status": "OK"}
        
    except Exception as e:
        # Al imprimir con flush=True, el error aparecerá al instante en el archivo .log
        import traceback
        print(f"\n [ERROR FATAL] en semilla {seed}:", flush=True)
        traceback.print_exc()
        return {"seed": seed, "status": f"ERROR: {e}"}
        
    finally:
        print(f"Tiempo de finalización: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        # Restaurar stdout y cerrar archivo
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()

if __name__ == '__main__':
    print("==================================================")
    print("   ORQUESTADOR MULTIPROCESO - MULTISEMILLA SUMO")
    print(f" Detectados {multiprocessing.cpu_count()} hilos lógicos en tu CPU.")
    print("==================================================\n")

    # 1. Configurar los experimentos (6 semillas diferentes)
    NUM_PROCESOS = 6
    EPISODIOS_POR_PROCESO = 6000
    TIPO_AGENTE = "TD"          # "TD", "SARSA", "LQR", etc.
    
    # Lista de configuraciones asignadas a cada worker
    configuraciones = [
        {"worker_id": 0, "seed": 707, "episodios": EPISODIOS_POR_PROCESO, "tipo_agente": TIPO_AGENTE},
        {"worker_id": 1, "seed": 808, "episodios": EPISODIOS_POR_PROCESO, "tipo_agente": TIPO_AGENTE},
        {"worker_id": 2, "seed": 909, "episodios": EPISODIOS_POR_PROCESO, "tipo_agente": TIPO_AGENTE},
        {"worker_id": 3, "seed": 1010, "episodios": EPISODIOS_POR_PROCESO, "tipo_agente": TIPO_AGENTE},
        {"worker_id": 4, "seed": 1111, "episodios": EPISODIOS_POR_PROCESO, "tipo_agente": TIPO_AGENTE},
        {"worker_id": 5, "seed": 1212, "episodios": EPISODIOS_POR_PROCESO, "tipo_agente": TIPO_AGENTE},
    ]

    print(f"Lanzando {NUM_PROCESOS} procesos paralelos en los P-cores...")
    print("Revisa el avance en vivo abriendo los archivos en la carpeta 'output_logs/'\n")
    
    tiempo_inicio = time.time()

    # 2. Crear el Pool de procesos
    # Recomendado: maxtasksperchild=1 garantiza que si un worker termina, 
    # Python destruye el proceso por completo liberando cualquier memoria de libsumo.
    with multiprocessing.Pool(processes=NUM_PROCESOS, maxtasksperchild=1) as pool:
        resultados = pool.map(worker_entrenamiento, configuraciones)

    tiempo_fin = time.time()
    horas = (tiempo_fin - tiempo_inicio) / 3600.0
    
    print("\n==================================================")
    print(" TODOS LOS PROCESOS HAN TERMINADO")
    print(" Resumen de ejecución:") 
    for res in resultados:
        estado_icono = "[CHECK]" if res["status"] == "OK" else "Not OK"
        print(f"   └── Semilla {res['seed']}: {estado_icono} {res['status']}")
    print(f"\n   Tiempo total de ejecución: {horas:.2f} horas")
    print("==================================================")