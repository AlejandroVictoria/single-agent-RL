import pandas as pd
import numpy as np


class ActionLogger:
    def __init__(self):
        self.log_acciones = []
        
    def registrar_paso(self, episodio, paso, datos_continuos, estado_discreto, 
                       accion_idx, tiempo_holding, tipo_seleccion, q_val_previo):
        """
        Almacena el contexto completo de una decisión tomada por el agente.
        """
        registro = {
            "episodio": int(episodio),
            "paso": int(paso),
            "vehiculo_id": str(datos_continuos.get("vehiculo_id", "")),
            "parada_id": str(datos_continuos.get("parada_id", "")),
            # Datos Continuos
            "dist_lider_float": float(datos_continuos.get("dist_lider", 0.0)),
            "dist_seguidor_float": float(datos_continuos.get("dist_seguidor", 0.0)),
            "t_espera_float": float(datos_continuos.get("tiempo_espera_promedio", 0.0)),
            # Estado Discreto (Desempaquetado)
            "bin_hf": int(estado_discreto[0]),      # Headway Front (delantero)
            "bin_hb": int(estado_discreto[1]),      # Headway Back (trasero)
            "bin_espera": int(estado_discreto[2]),  # Nivel de espera en estación
            "es_terminal": int(0),               # Reemplaza 'done' por la variable que controle tu estado terminal
            # Acción y Origen
            "accion_idx": int(accion_idx),
            "holding_seg": float(tiempo_holding),
            "origen": str(tipo_seleccion), # 'exploracion' o 'politica'
            # Métrica Q
            "q_val_previo": float(q_val_previo)
        }
        self.log_acciones.append(registro)
        
    def registrar_recompensa_y_update(self, recompensa, q_val_nuevo):
        """
        Actualiza el último registro con la retroalimentación recibida del entorno.
        """
        if self.log_acciones:
            self.log_acciones[-1]["recompensa"] = float(recompensa)
            self.log_acciones[-1]["q_val_nuevo"] = float(q_val_nuevo)
            self.log_acciones[-1]["delta_q"] = float(q_val_nuevo - self.log_acciones[-1]["q_val_previo"])

    def obtener_resumen_episodio(self, episodio, recompensa_acumulada, epsilon, alpha):
        """
        Calcula la distribución de acciones y ratio de exploración para la consola.
        """
        df_ep = pd.DataFrame(self.log_acciones)
        df_ep = df_ep[df_ep["episodio"] == episodio]
        
        if df_ep.empty:
            return "No hay registros para este episodio."
            
        conteo_acciones = df_ep["holding_seg"].value_counts(normalize=True).to_dict()
        ratio_exp = (df_ep["origen"] == "exploracion").mean()
        rec_total = df_ep["recompensa"].sum()
        # epsilon = float(epsilon)
        # alpha = float(alpha)
        
        resumen = f"--- Resumen Ep {episodio} (Rec Total: {rec_total:.2f}) ---\n"
        resumen += f"  > Ratio Exploración: {ratio_exp*100:.1f}%\n"
        resumen += f"  > Distribución de Retenciones: "
        for seg, pct in sorted(conteo_acciones.items()):
            resumen += f"[{int(seg)}s: {pct*100:.1f}%] "
        resumen += f"\n"
        resumen += f"  > Recompensa: {recompensa_acumulada:.2f} | Epsilon: {epsilon:.3f} | Alpha: {alpha:.3f}"
        return resumen

    def exportar_parquet(self, ruta_archivo="logs_telemetria_acciones.parquet"):
        """
        Guarda todo el historial en un archivo Parquet para análisis rápido.
        """
        df = pd.DataFrame(self.log_acciones)
        df.to_parquet(ruta_archivo, index=False)
        print(f"  > Seguimiento a acciones tomadas exportado exitosamente a: {ruta_archivo}")
        
    def limpiar_memoria(self):
        self.log_acciones.clear()