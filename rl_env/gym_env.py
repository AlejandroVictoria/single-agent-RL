# rl_env/gym_env.py
import gymnasium as gym
import numpy as np
import csv
import os
import sys

from sumo_manager.sumo_manager import SumoManager
import libsumo as traci


class ZacatencoTrolebusEnv(gym.Env):
    def __init__(self, config_path="sumo_files/configuracion/zacatenco_v4.sumocfg", grabar_datos=False, **kwargs):
        super(ZacatencoTrolebusEnv, self).__init__()
        dir_archivo_actual = os.path.dirname(os.path.abspath(__file__))
        dir_exp = os.path.dirname(dir_archivo_actual)
        dir_raiz = os.path.dirname(dir_exp)
        # ruta_sumocfg = os.path.join(dir_raiz, "sumo_files", "configuracion", "zacatenco_v4.sumocfg")
        ruta_sumocfg = os.path.join("sumo_files", "configuracion", "zacatenco_v4.sumocfg")

        self.sumo = SumoManager(ruta_sumocfg)
        self.tiempo_limite = kwargs['tiempo_limite']
        # Variables de control interno
        self.grabar_datos = grabar_datos
        self.historial_telemetria = []
        self.episodio_actual = 0
        self.pending_transitions = {}
        self.last_vehiculo_id = None
        
        # Espacios de Gymnasium
        self.n_actions = kwargs['n_acciones']
        self.shape_state = list(kwargs['espacio_estados'])

        self.action_space = gym.spaces.Discrete(self.n_actions)
        # Asumiendo 3 niveles de dist_lider, 3 dist_seguidor, 3 de peatones
        self.observation_space = gym.spaces.MultiDiscrete(self.shape_state)
        # Control trolebuses
        self.bloqueo_hasta = 0
        self.bloqueo_vehiculos = {}
        self.queue_estados = []
        self.puntos_regulacion = ["bs_1", "bs_19", "bs_28"]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sumo.start(seed=seed)
        
        self.experimento_actual = options.get("experimento_id", "default_exp") if options else "default_exp"
        self.episodio_actual = options.get("episodio", 0) if options else 0
        e_folder = options.get("e_folder", None)
        # self.episodio_actual += 1

        self.base_path = os.path.join("data", self.experimento_actual)
        os.makedirs(os.path.join(self.base_path, "trolebuses"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "paradas"), exist_ok=True)

        # Reiniciar contadores para el nuevo episodio
        self.sumo.reset_historial()

        
        # --- LIMPIEZA DE MEMORIA (Blindaje de Episodio) ---
        self.bloqueo_vehiculos = {}       
        self.last_vehiculo_id = None
        self.last_estacion_id = None        
        
        # --- BUCLE DE AVANCE RÁPIDO PARA EL PRIMER ESTADO ---
        obs, data, tiempo_simulacion = self._event_manager()
        self.last_obs = obs
        # print(f"self.last_obs: {obs}")
        info = {
            "datos_continuos": data
        }
        return obs, info

    def step(self, action):
        """Aplica la retención en puntos clave y avanza síncronamente ignorando paradas ordinarias."""
        tiempo_actual = traci.simulation.getTime()

        # print(f"Esta es la lista de vehículos bloqueados: {self.bloqueo_vehiculos} | Tiempo actual: {tiempo_actual}")
        for vehiculo_id in list(self.bloqueo_vehiculos.keys()):
            bloqueo_info = self.bloqueo_vehiculos[vehiculo_id]
            bloqueo_expira = bloqueo_info.get("bloqueo_expira")
            
            if bloqueo_expira is not None:
                if tiempo_actual >= bloqueo_expira:
                    # print(f"DEBUG: Expirando bloqueo de {vehiculo_id} en {bloqueo_info['estacion_bloqueo']} a tiempo {tiempo_actual}")
                    del self.bloqueo_vehiculos[vehiculo_id]

        # print(f"DEBUG: Estado de lista de vehículos bloqueados antes de aplicar acción: {self.bloqueo_vehiculos}")
        # print(f"DEBUG: self.last_vehiculo_id={self.last_vehiculo_id}, self.last_estacion_id={self.last_estacion_id}, tiempo_actual={tiempo_actual}, acción={action}")
        # 1. APLICAR ACCIÓN AL VEHÍCULO DEL ESTADO ANTERIOR (Bloqueo Individual)
        if self.last_vehiculo_id is not None:
            veh_actual = self.last_vehiculo_id
            estacion_actual = self.last_estacion_id

            if veh_actual in self.pending_transitions:
                self.pending_transitions[veh_actual]["accion"] = action
            else:
                self.pending_transitions[veh_actual] = {
                    "estado_previo": self.last_obs, # Se guarda el estado que originó la decisión.
                    "accion": action
                }

            if veh_actual in self.bloqueo_vehiculos and not self.bloqueo_vehiculos[veh_actual]["accion aplicada"]:
                tiempo_retencion = action * 15
                tiempo_base_estructural = 120 if estacion_actual == "bs_28" else 0
                tiempo_total_bloqueo = tiempo_base_estructural + tiempo_retencion
                
                self.sumo.aplicar_accion(veh_actual, tiempo_total_bloqueo)
                self.bloqueo_vehiculos[veh_actual]["accion aplicada"] = True
                tiempo_llegada = self.bloqueo_vehiculos[veh_actual]["tiempo_llegada"]
                self.bloqueo_vehiculos[veh_actual]["tiempo_bloqueo"] = tiempo_total_bloqueo
                self.bloqueo_vehiculos[veh_actual]["bloqueo_expira"] = tiempo_llegada + tiempo_total_bloqueo + 10

        # 2. BUCLE DE AVANCE RÁPIDO (Filtro de Puntos de Regulación)        
        obs, datos_continuos, tiempo_simulacion = self._event_manager()

        if tiempo_simulacion >= self.tiempo_limite:
            # El episodio terminó por tiempo
            return obs, 0.0, True, False, {"msg": "Horizonte completado.", "tiempo": tiempo_simulacion}

        # 5. RECOMPENSA
        reward = self._reward_function(datos_continuos)

        # Se guarda el estado actual para que en el próximo ste() sepamos qué observación generó la acción
        self.last_obs = obs

        # EN LUGAR DE SOLO DEVOLVER INFORMACIÓN SUELTA, DEVOLVEMOS EL ID DEL VEHÍCULO EN INFO
        info = {
            "tiempo": tiempo_simulacion,
            "vehiculo_id": self.last_vehiculo_id,
            "estacion_id": self.last_estacion_id,
            "datos_continuos": datos_continuos
        }

        return obs, reward, False, False, info
    
    def _event_manager(self):
        """Avanza la simulación hasta el siguiente evento de regulación y devuelve el estado discreto."""
        if len(self.queue_estados) > 0:
            obs_pre = self.queue_estados.pop()
            obs = obs_pre[0]
            self.last_vehiculo_id = obs_pre[1]
            self.last_estacion_id = obs_pre[2]
            datos_continuos = obs_pre[3]
            tiempo_actual = traci.simulation.getTime()

            return obs, datos_continuos, tiempo_actual
        
        # --- BUCLE DE AVANCE RÁPIDO PARA EL PRIMER ESTADO ---
        raw_data = None
        terminated = False
        break_eventos = True

        while not terminated and break_eventos:
            # 0. Validar vehículos anteriormente bloqueados y limpiar los que ya expiraron
            raw_data, terminated = self.sumo.avanzar_hasta_evento(tiempo_limite=self.tiempo_limite)
                
            if terminated or raw_data is None:
                # Creamos variables ficticias (dummy) seguras
                obs_dummy = np.array([2, 2, 0], dtype=np.int32)
                datos_dummy = {
                    "peatones": 0, 
                    "dist_lider": -1.0, 
                    "dist_seguidor": -1.0, 
                    "tiempo_simulacion": self.tiempo_limite
                }
                # Forzamos la salida devolviendo el tiempo límite para que step() lo corte
                return obs_dummy, datos_dummy, self.tiempo_limite
            
            # forma de raw_data: 
            # {'raw_data': 
            #   [(vehiculo_id, estacion_id),(vehiculo_id, estacion_id), ...],
            # "timestamp": 
            #   tiempo_actual}
            tiempo_actual = raw_data.get('timestamp', 0)
            lista_eventos = raw_data.get('raw_data', [])

            for evento in lista_eventos:
                vehiculo_actual,estacion_actual = evento[0],evento[1]

                if estacion_actual in self.puntos_regulacion:
                    if not self.bloqueo_vehiculos.get(vehiculo_actual):
                        # print(f"DEBUG: Registrando bloqueo para {vehiculo_actual} en {estacion_actual} a tiempo {tiempo_actual}")

                        self.bloqueo_vehiculos[vehiculo_actual] = {
                            "estacion_bloqueo": estacion_actual,
                            "tiempo_llegada": tiempo_actual,
                            "accion aplicada": False
                            }
                    
                        datos_continuos = self.sumo.get_data_from_vehicle(vehiculo_actual, estacion_actual)
                        estado = self._discretizar_datos(
                            datos_continuos["parada_id"],
                            datos_continuos["dist_lider"],
                            datos_continuos["dist_seguidor"],
                            datos_continuos["tiempo_espera_promedio"]
                        )

                        self.queue_estados.append([estado, vehiculo_actual, estacion_actual, datos_continuos])
                        break_eventos = False                      

        # print(f"DEBUG: Salida de _event_manager() con {len(self.queue_estados)} estados en cola y tiempo actual {tiempo_actual}")
        if len(self.queue_estados) != 0:
            obs_pre = self.queue_estados.pop()
            obs = obs_pre[0]
            self.last_vehiculo_id = obs_pre[1]
            self.last_estacion_id = obs_pre[2]
            datos_continuos = obs_pre[3]
        
        # print(f"DEBUG: Estado retornado por _event_manager(): {obs}, tiempo_actual={tiempo_actual}")
        return obs, datos_continuos, tiempo_actual
    
    def _discretizar_datos(
        self, estacion_id_str, dist_lider, dist_seguidor, tiempo_espera_promedio
    ):
        """
        Transforma la telemetría espacial directa de TraCI en un espacio discreto 
        de 196 estados (7 x 7 x 4). Utiliza umbrales fijos centrados en un headway 
        ideal (tau) de 1600 metros para la parte local, y el Coeficiente de Variación 
        (CV) global como "Termómetro del Caos" (Experimento 4).
        """
        tau = 1600.0
        
        # 1. Sanitización de datos entrantes locales
        # La función get_data_from_vehicle asigna -1 cuando no hay líder o seguidor.
        # En ese caso, le decimos al agente que asuma el estado ideal (1600m) para
        # que no tome acciones agresivas intentando corregir "fantasmas".
        d_lider = (
            float(dist_lider)
            if (dist_lider is not None and float(dist_lider) != -1.0)
            else tau
        )
        d_seguidor = (
            float(dist_seguidor)
            if (dist_seguidor is not None and float(dist_seguidor) != -1.0)
            else tau
        )
        # -------------------------------------------------------------------------
        # DIMENSIÓN 1 y 2: Headway Espacial [0, 6] -> Ideal tau = 1600m
        # Ancho de banda de 300m en zonas medias para sincronizar con la máxima
        # acción de holding de 60s (asumiendo ~5m/s comercial).
        # -------------------------------------------------------------------------
        bins_espaciales = [
            800.0,   # 0: < 800m -> Bunching Crítico (Colisión inminente)
            1150.0,  # 1: 800 - 1150m -> Déficit Severo
            1450.0,  # 2: 1150 - 1450m -> Déficit Moderado
            1750.0,  # 3: 1450 - 1750m -> ZONA IDEAL (Centrado en tau=1600m)
            2050.0,  # 4: 1750 - 2050m -> Exceso Moderado
            2400.0,  # 5: 2050 - 2400m -> Exceso Severo
                     # 6: > 2400m -> Brecha Crítica
        ]
        
        estado_hf = int(np.digitize(d_lider, bins_espaciales))
        estado_hb = int(np.digitize(d_seguidor, bins_espaciales))

        # -------------------------------------------------------------------------
        # DIMENSIÓN 3: Termómetro del Caos (CV Global) [0, 3] -> 4 niveles
        # -------------------------------------------------------------------------
        headways, _ = self.sumo.get_headways_global(tau=tau)
        
        if len(headways) >= 2:
            media_hw = np.mean(headways)
            cv_actual = np.std(headways) / media_hw if media_hw > 0 else 0.0
        else:
            cv_actual = 0.0
            
        bins_cv = [
            0.25,  # 0: CV < 0.25 -> Operación ideal / Reloj
            0.50,  # 1: 0.25 a 0.50 -> Irregularidad leve
            0.75   # 2: 0.50 a 0.75 -> Bus Bunching en formación
                   # 3: CV >= 0.75 -> Caos severo / Pelotón consolidado
        ]
        estado_cv = int(np.digitize(cv_actual, bins_cv))

        # Vector resultante listo para ser tupla clave en la Q-Table
        return np.array([estado_hf, estado_hb, estado_cv], dtype=np.int32)

    def _reward_function(self, datos_continuos):
        """
        Adaptación de la recompensa de Zheng et al. para espacio métrico.
        Corregida para el escalado en metros y la protección de bordes de flotilla.
        """
        tau = 1600.0   # Headway objetivo en metros
        delta = 800.0  # Umbral de desviación (Alineado con bines 0 y 6)
        epsilon = 1e-6 
        
        # 1. Extracción segura: Si viene -1 (sin líder/seguidor), asumimos el estado ideal
        # para no castigar injustamente a los vehículos en los extremos de la línea.
        dist_lider_cruda = datos_continuos.get("dist_lider", tau)
        dist_seg_cruda = datos_continuos.get("dist_seguidor", tau)
        
        h_f = float(tau if dist_lider_cruda == -1.0 else dist_lider_cruda)
        h_b = float(tau if dist_seg_cruda == -1.0 else dist_seg_cruda)
        
        # Cálculo de los errores absolutos |h - tau|
        err_f = abs(h_f - tau)
        err_b = abs(h_b - tau)
        
        # 2. Schedule alignment: phi(h) = -|h - tau|
        phi_f = -err_f
        phi_b = -err_b
        
        # 3. Pesos dinámicos: omega(hf, hb)
        omega = err_f / (err_f + err_b + epsilon)
        
        # 4. Headway symmetry: Penalización por asimetría espacial
        penalizacion_simetria = 0.5 * abs(h_f - h_b)
        
        # 5. Robust penalization adaptada
        # Si la máxima acción de holding es 60s (~300m de castigo auto-infligido para corregir),
        # cruzar el umbral delta debe doler mucho más que cualquier holding.
        # Elevamos la magnitud a 2000 para que actúe como un verdadero "muro" negativo.
        penalizacion_robusta = 0.0
        if err_f > delta or err_b > delta:
            penalizacion_robusta = 2000.0  # Escalado
            
        # 6. Superficie de recompensa final R(hf, hb)
        recompensa_final = (omega * phi_f) + ((1.0 - omega) * phi_b) - penalizacion_simetria - penalizacion_robusta
        
        return float(recompensa_final)

    @property
    def grabar_datos(self):
        return self.sumo.grabar_datos

    @grabar_datos.setter
    def grabar_datos(self, valor: bool):
        # Cuando agent_training dice: env.unwrapped.grabar_datos = True
        # el entorno se lo pasa directamente al motor SUMO:
        self.sumo.grabar_datos = valor

    def _volcar_a_csv(self, exp_folder):
        # El entorno delega el trabajo pesado a sumo_manager:
        self.sumo._volcar_a_csv(self.experimento_actual, self.episodio_actual, exp_folder)

    def close(self):
        # Aseguramos liberar los binarios al cerrar el script de entrenamiento
        self.sumo.stop()