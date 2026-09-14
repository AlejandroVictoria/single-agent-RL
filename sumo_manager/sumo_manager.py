import libsumo as traci
#import traci
import sys
import os
import csv


class SumoManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.grabar_datos = False
        self.historial_troles = []
        self.historial_paradas = []
        self.intervalo_telemetria = 15
        # libsumo requiere obligatoriamente el binario en modo consola
        self.sumo_binary = "sumo"
        #self.sumo_binary = "sumo-gui"  # Para depuración visual

    def reset_historial(self):
        self.historial_paradas = []
        self.historial_troles = []

    def start(self, seed=None):
        """Inicializa el motor con escala de tráfico controlada."""
        self.stop()

        # Agregamos "--scale", "0.7" al comando
        cmd = [
            self.sumo_binary,
            "-c", self.config_path,
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--no-warnings", "true",
            "--verbose", "false",
            "--scale", "0.3"  # Reducción al 70% del tráfico original
        ]
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
            
        traci.start(cmd)

    def step(self):
        """Avanza un paso discreto de tiempo en la simulación de SUMO."""
        traci.simulationStep()

    def stop(self):
        """Libera de forma segura el proceso del simulador de la memoria RAM."""
        try:
            traci.close()
            sys.stdout.flush()
        except Exception:
            pass

    def aplicar_accion(self, vehiculo_id, holding_seconds):
        TIEMPOS_BASE = {
            "bs_28": 0,
            "bs_1": 0,
            "bs_19": 0
        }

        # Si el agente dice 0s, ignoramos.
        if holding_seconds <= 0:
            return
        try:
            # Obtenemos el estado actual de la parada en SUMO
            stops = traci.vehicle.getStops(vehiculo_id, limit=1)
            
            tiempo_base = TIEMPOS_BASE.get(stops[0].stoppingPlaceID, 0)
            espera_total = holding_seconds + tiempo_base
            
            if stops:
            #if stops[0].stoppingPlaceID in TIEMPOS_BASE.keys():
                # print(f"DEBUG: Vehículo {vehiculo_id} en parada {stops[0].stoppingPlaceID} con duración {stops[0].duration} y espera total {espera_total}s")
                # COMPROBACIÓN CRÍTICA:
                # Solo actualizamos si la duración es diferente a la que ya tiene asignada.
                # Esto evita el reinicio del cronómetro en cada iteración.
                if stops[0].duration != espera_total:
                    traci.vehicle.setBusStop(
                        vehID=vehiculo_id, 
                        stopID=stops[0].stoppingPlaceID, 
                        duration=float(espera_total),
                        flags=0
                    )
            
        except traci.exceptions.TraCIException:
            pass

    def avanzar_hasta_evento(self, tiempo_limite=10800):
        """
        Avanza la simulación paso a paso verificando eventos concurrentes ANTES
        de mover el reloj, evitando saltarse trolebuses con tiempos de abordaje cortos.
        """
        terminated = False
        trolebuses_evento = None

        while not terminated:
            self.step()

            trolebuses_evento = self._detectar_evento()
            tiempo_actual = traci.simulation.getTime()
            
            if self.grabar_datos and (tiempo_actual % self.intervalo_telemetria == 0):
                troles_globales, paradas_globales = self.get_telemetria_global()
                self.historial_troles.extend(troles_globales)
                self.historial_paradas.extend(paradas_globales)

            if len(trolebuses_evento) > 0:
                break

            # print(f"DEBUG: Trolebuses: {trolebuses_evento} | timestamp {tiempo_actual}")
                
            if tiempo_actual >= tiempo_limite:
                terminated = True
                break
                
        # Si el episodio terminó por tiempo
        if trolebuses_evento is None and terminated:
            return None, True
        
        return {'raw_data':trolebuses_evento, "timestamp": tiempo_actual}, terminated

    def _detectar_evento(self):
        """
        Iterador seguro que detecta vehículos detenidos y evita reportar 
        el mismo vehículo dos veces en la misma parada o bloquearse por falsos positivos.
        """
        vehiculos_actuales = self._extraer_troles(traci.vehicle.getIDList())
        trolebuses_evento = []

        for veh_id in vehiculos_actuales:
            if "trolebus" in veh_id: 
                is_stopped = traci.vehicle.isStopped(veh_id)

                if is_stopped:
                    estacion_id = traci.vehicle.getStops(veh_id, limit=1)
                    estacion_id = estacion_id[0].stoppingPlaceID

                    if estacion_id:
                        trolebuses_evento.append((veh_id,estacion_id))
        
        return trolebuses_evento
    
    def _extraer_troles(self, lista_vehiculos):
        """
        Filtra la lista de vehículos para devolver solo los trolebuses.
        """
        len_tuple = len(lista_vehiculos)
        
        if len_tuple == 0:
            return []
        elif len_tuple == 1:
            return [lista_vehiculos[0]] if "trolebus" in lista_vehiculos[0].lower() else []
        else:
            lista_troles = []
            index = -1

            while "trolebus" in lista_vehiculos[index]:
                lista_troles.append(lista_vehiculos[index])
                index -= 1
            return lista_troles

    def get_data_from_vehicle(self, vehiculo_id, estacion_id):
        """
        Extrae la telemetría necesaria para el vector de estado (Q-Learning)
        y para el volcado de datos de la tesis (CSV).
        """
        tiempo_actual = traci.simulation.getTime()
        
        # --- 1. DATOS DE LA ESTACIÓN ---
        personas_en_parada = traci.busstop.getPersonIDs(estacion_id)
        peatones_esperando = len(personas_en_parada)

        tiempo_espera_promedio = 0.0
        if peatones_esperando > 0:
            # 2. Sumar el tiempo individual de espera de cada peatón en el andén
            tiempos_espera = [
                traci.person.getWaitingTime(p_id) for p_id in personas_en_parada
            ]
            tiempo_espera_promedio = float(sum(tiempos_espera) / peatones_esperando)

        # --- 2. CÁLCULO DE DISTANCIAS 1D (ODOMETRÍA) ---
        dist_lider = None
        dist_seguidor = -1.0
        
        # Generar trolebuses lider y seguidor
        name_trolebus,id_trolebus = vehiculo_id.split(".")
        id_trolebus = int(id_trolebus)
        if id_trolebus != 0:
            id_trolebus_lider = id_trolebus - 1 if id_trolebus != 0 else -1
            trolebus_lider = name_trolebus + "." + str(id_trolebus_lider)
        else:
            trolebus_lider = None
            
        id_trolebus_seguidor = id_trolebus + 1
        trolebus_seguidor = name_trolebus + "." + str(id_trolebus_seguidor)
        trolebuses = [trolebus_lider, vehiculo_id, trolebus_seguidor]

        # Extraer las distancias y organizar la flotilla
        distancias_troles = []
        for t_id in trolebuses:
            try:
                if t_id is not None:
                    dist = traci.vehicle.getDistance(t_id)
                else:
                    dist = None
            # except traci.exceptions.TraCIException as e:
            except traci.libsumo.TraCIException as e:
                if "is not known." in str(e):
                    dist = None
                
            distancias_troles.append((t_id, dist))

        dist_actual = distancias_troles[1]
        dist_lider = distancias_troles[0]
        dist_seguidor = distancias_troles[2]

        if dist_lider[1] != None:
            dist_lider = dist_lider[1] - dist_actual[1]
        else:
            dist_lider = -1
        
        if dist_seguidor[1] != None:
            dist_seguidor = dist_actual[1] - dist_seguidor[1]
        else:
            dist_seguidor = -1

        return {
            "vehiculo_id": vehiculo_id,
            "parada_id": estacion_id,
            "dist_lider": dist_lider,
            "dist_seguidor": dist_seguidor,
            "peatones": peatones_esperando,
            "tiempo_espera_promedio": tiempo_espera_promedio,
            "tiempo_simulacion": tiempo_actual,
        }
    
    def get_telemetria_global(self):
        """
        Toma una fotografía de TODA la red para los CSV de la tesis.
        Independiente del vector de estado del agente.
        """
        tiempo_actual = traci.simulation.getTime()
        datos_troles = []
        datos_paradas = []
        
        # 1. Escanear TODOS los vehículos en el mapa
        for v_id in traci.vehicle.getIDList():
            if "trolebus" in v_id.lower():  # Filtrar para no agarrar autos normales
                #print("Erroir en get_telemetria_global")
                x, y = traci.vehicle.getPosition(v_id)
                lon, lat = traci.simulation.convertGeo(x, y, fromGeo=False)
                distancia = traci.vehicle.getDistance(v_id)
                pasajeros = traci.vehicle.getPersonNumber(v_id)
                velocidad = traci.vehicle.getSpeed(v_id) * 3.6
                
                # Tratar de obtener la parada actual o próxima
                try:
                    paradas = traci.vehicle.getStops(v_id, limit=1)
                    prox_parada = paradas[0].stoppingPlaceID if paradas else "en_ruta"
                except:
                    prox_parada = "desconocida"
                
                datos_troles.append({
                    "tiempo_simulacion": tiempo_actual,
                    "trolebus_id": v_id,
                    "proxima_estacion": prox_parada,
                    "pasajeros_a_bordo": pasajeros,
                    "velocidad_kmh": round(velocidad, 2),
                    "distancia_recorrida_m": round(distancia, 2),
                    "latitud": lat,
                    "longitud": lon
                })
        
        # 2. Escanear TODAS las paradas en el mapa
        for p_id in traci.busstop.getIDList():
            peatones = traci.busstop.getPersonCount(p_id)
            
            try:
                nombre = traci.busstop.getName(p_id)
            except:
                nombre = p_id
                
            tiempo_espera = 0
            try:
                personas = traci.busstop.getPersonIDs(p_id)
                for p in personas:
                    tiempo_espera += traci.person.getWaitingTime(p)
            except:
                pass
                
            datos_paradas.append({
                "tiempo_simulacion": tiempo_actual,
                "estacion_id": p_id,
                "nombre_parada": nombre,
                "personas_esperando": peatones,
                "tiempo_espera_acumulado_s": tiempo_espera
            })
            
        return datos_troles, datos_paradas

    def get_headways_global(self, tau=1600):
        """
        Extrae la distancia recorrida de todos los trolebuses activos 
        y calcula los headways espaciales instantáneos de la línea.
        En los bordes (vehículo sin líder o sin seguidor), inserta 
        la distancia ideal (tau) por defecto.
        """
        distancias_troles = []
        
        # 1. Extraer identificadores (corrigiendo la llamada a TraCI)
        lista_troles = self._extraer_troles(traci.vehicle.getIDList())

        for v_id in lista_troles:
            distancia = traci.vehicle.getDistance(v_id)
            distancias_troles.append(distancia)
        
        # 2. Ordenar las distancias de mayor a menor 
        # (el vehículo con mayor distancia recorrida va liderando la línea)
        distancias_troles.sort(reverse=True)
        
        headways = []
        
        # Si la red está vacía, retornar listas vacías inmediatamente
        if not distancias_troles:
            return headways, distancias_troles
            
        # 3. Borde delantero: El primer vehículo no tiene líder.
        # Asumimos que su headway frontal es perfecto (tau).
        headways.append(tau)
        
        # 4. Headways internos: Diferencia espacial real entre vehículos consecutivos.
        for i in range(len(distancias_troles) - 1):
            headway = distancias_troles[i] - distancias_troles[i+1]
            headways.append(headway)
            
        # 5. Borde trasero: El último vehículo no tiene seguidor.
        # Asumimos que el espacio hacia atrás es perfecto (tau).
        headways.append(tau)
                
        return headways, distancias_troles

    def get_headways_global(self, tau=1600):
            """
            Extrae la distancia recorrida de todos los trolebuses activos 
            y calcula los headways espaciales instantáneos de la línea.
            En los bordes (vehículo sin líder o sin seguidor), inserta 
            la distancia ideal (tau) por defecto.
            """
            distancias_troles = []
            
            # 1. Extraer identificadores (corrigiendo la llamada a TraCI)
            lista_troles = self._extraer_troles(traci.vehicle.getIDList())
    
            for v_id in lista_troles:
                distancia = traci.vehicle.getDistance(v_id)
                distancias_troles.append(distancia)
            
            # 2. Ordenar las distancias de mayor a menor 
            # (el vehículo con mayor distancia recorrida va liderando la línea)
            distancias_troles.sort(reverse=True)
            
            headways = []
            
            # Si la red está vacía, retornar listas vacías inmediatamente
            if not distancias_troles:
                return headways, distancias_troles
                
            # 3. Borde delantero: El primer vehículo no tiene líder.
            # Asumimos que su headway frontal es perfecto (tau).
            headways.append(tau)
            
            # 4. Headways internos: Diferencia espacial real entre vehículos consecutivos.
            for i in range(len(distancias_troles) - 1):
                headway = distancias_troles[i] - distancias_troles[i+1]
                headways.append(headway)
                
            # 5. Borde trasero: El último vehículo no tiene seguidor.
            # Asumimos que el espacio hacia atrás es perfecto (tau).
            headways.append(tau)
                    
            return headways, distancias_troles
    
    def _volcar_a_csv(self, experimento_actual, episodio_actual, exp_folder):
        if not self.historial_troles and not self.historial_paradas:
            return
        # 1. Crear directorios exactos
        base_path = f"data/{experimento_actual}"
        path_troles = f"{base_path}/trolebuses"
        path_paradas = f"{base_path}/paradas"
        os.makedirs(path_troles, exist_ok=True)
        os.makedirs(path_paradas, exist_ok=True)

        ruta_t = f"{path_troles}/ep_{episodio_actual:03d}.csv"
        ruta_p = f"{path_paradas}/ep_{episodio_actual:03d}.csv"

        # 2. Guardar Trolebuses
        cabeceras_t = ["tiempo_simulacion", "trolebus_id", "proxima_estacion", "pasajeros_a_bordo", "velocidad_kmh", "distancia_recorrida_m", "latitud", "longitud"]

        with open(ruta_t, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cabeceras_t)
            writer.writeheader()
            writer.writerows(self.historial_troles)

        # 3. Guardar Paradas
        cabeceras_p = ["tiempo_simulacion", "estacion_id", "nombre_parada", "personas_esperando", "tiempo_espera_acumulado_s"]
        with open(ruta_p, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cabeceras_p)
            writer.writeheader()
            writer.writerows(self.historial_paradas)

        print(f"  > Historial del episodio {episodio_actual} exportado a {base_path}")