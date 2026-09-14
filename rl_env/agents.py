import numpy as np

class BaseAgent:
    """
    Clase base abstracta para los agentes de Aprendizaje por Refuerzo.
    """
    def __init__(self, state_shape, n_actions, gamma=0.99, **kwargs):
        self.state_shape = state_shape
        self.n_actions = n_actions

        alpha = kwargs["alpha"]
        epsilon = kwargs["epsilon"]

        self.gamma = gamma

        self.alpha = alpha["alpha"]
        self.min_alpha = alpha["min"]
        self.alpha_decay_rate = alpha["decay"]
        
        self.epsilon = epsilon["epsilon"]
        self.min_epsilon = epsilon["min"]
        self.epsilon_decay_rate = epsilon["decay"]

        # Inicialización de la matriz de conocimiento (Q-Table) N-Dimensional
        self.q_table = np.zeros((*state_shape, n_actions))

    def elegir_accion(self, estado):
        """Selecciona una acción utilizando la política epsilon-greedy."""
        estado_tupla = tuple(np.array(estado).flatten())
        q_valores_estado = self.q_table[estado_tupla]

        if np.random.rand() < self.epsilon:
            accion_idx = np.random.randint(self.n_actions)
            tipo_sel = "exploracion"
        else:
            max_val = np.max(q_valores_estado)
            indices_maximos = np.flatnonzero(q_valores_estado == max_val)
            accion_idx = np.random.choice(indices_maximos)
            tipo_sel = "politica"
        
        q_previo = float(q_valores_estado[accion_idx])

        return accion_idx, tipo_sel, q_previo

    def decaer_epsilon(self):
        """Reduce el factor de exploración gradualmente de forma segura."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)

    def decaer_alpha(self):
        """Reduce el factor de exploración gradualmente de forma segura."""
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay_rate)

    def actualizar_paso(self, estado, accion, recompensa, siguiente_estado, terminado):
        raise NotImplementedError

    def actualizar_fin_episodio(self):
        pass


class QLearningAgent(BaseAgent):
    """
    Agente de Diferencias Temporales (TD) - Variante Q-Learning.
    """
    def __init__(self, state_shape, n_actions, **kwargs):
        super().__init__(state_shape, n_actions, **kwargs)

    def actualizar_paso(self, estado, accion, recompensa, siguiente_estado, terminado):
        # 1. Aseguramos que 'estado' sea una tupla de enteros
        # Si estado es un array (como [[5 0 0 0]]), se aplana:
        idx_estado = tuple(np.array(estado).flatten())
        
        # 2. Ahora usamos esa tupla para acceder
        valor_actual = self.q_table[idx_estado][accion]
        
        max_futuro = 0 if terminado else np.max(self.q_table[tuple(np.array(siguiente_estado).flatten())])
        target = recompensa + self.gamma * max_futuro
        
        # 3. Guardamos el cambio
        q_nuevo = valor_actual + self.alpha * (target - valor_actual)
        self.q_table[idx_estado][accion] = q_nuevo
        return q_nuevo
        
    def actualizar_fin_episodio(self):
        self.decaer_epsilon()
        self.decaer_alpha()


class MCAgent(BaseAgent):
    """
    Agente Monte Carlo (MC) - Variante Every-Visit.
    """
    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__(state_shape, n_actions, alpha, gamma, epsilon, epsilon_decay, min_epsilon)
        self.trayectoria = []  

    def actualizar_paso(self, estado, accion, recompensa, siguiente_estado, terminado):
        # MC solo guarda la memoria del paso, no actualiza la tabla todavía
        self.trayectoria.append((estado, accion, recompensa))

    def actualizar_fin_episodio(self, gamma=None):
        # Usar el gamma del argumento si se pasa, sino el de la clase
        g_gamma = gamma if gamma is not None else self.gamma
        G = 0

        # Recorremos la historia desde el final hacia el principio
        for estado, accion, recompensa in reversed(self.trayectoria):
            G = recompensa + g_gamma * G
            
            idx_estado = tuple(np.array(estado).flatten())
            
            # Actualizamos la celda usando el índice normalizado
            self.q_table[idx_estado][accion] += self.alpha * (G - self.q_table[idx_estado][accion])
            
        # Limpiamos la memoria para el próximo episodio
        self.trayectoria = []
        self.decaer_epsilon()
        print(f"DEBUG MC: Fin de actualización. Epsilon ahora es: {self.epsilon}") 


class ExpectedSARSAAgent(BaseAgent):
    """
    Agente Expected SARSA.
    Calcula el target basándose en la expectativa del valor del siguiente estado
    bajo una política epsilon-greedy, reduciendo la varianza respecto a SARSA tradicional.
    """
    def __init__(self, state_shape, n_actions, **kwargs):
        super().__init__(state_shape, n_actions, **kwargs)

    def actualizar_paso(self, estado, accion, recompensa, siguiente_estado, terminado):
        # 1. Aseguramos que los estados sean tuplas de enteros (aplanados)
        idx_estado = tuple(np.array(estado).flatten())
        idx_siguiente_estado = tuple(np.array(siguiente_estado).flatten())
        
        # 2. Obtenemos el valor Q actual
        valor_actual = self.q_table[idx_estado][accion]
        
        # 3. Calculamos el valor esperado del siguiente estado
        if terminado:
            valor_esperado_futuro = 0.0
        else:
            q_siguiente = self.q_table[idx_siguiente_estado]
            max_q_siguiente = np.max(q_siguiente)
            suma_q_siguiente = np.sum(q_siguiente)
            
            # La expectativa bajo la política epsilon-greedy se calcula como:
            # (Probabilidad de explorar * Promedio de todos los valores Q) + 
            # (Probabilidad de explotar * Valor Q máximo)
            valor_esperado_futuro = (self.epsilon / self.n_actions) * suma_q_siguiente + \
                                    (1.0 - self.epsilon) * max_q_siguiente
            
        # 4. Calculamos el target y la actualización (similar a Q-Learning)
        target = recompensa + self.gamma * valor_esperado_futuro
        q_nuevo = valor_actual + self.alpha * (target - valor_actual)
        
        # 5. Guardamos el cambio
        self.q_table[idx_estado][accion] = q_nuevo
        
        return q_nuevo
        
    def actualizar_fin_episodio(self):
        self.decaer_epsilon()
        self.decaer_alpha()


class SARSAAgent(BaseAgent):
    """
    Agente de Diferencias Temporales (TD) - Variante SARSA (On-policy).
    """
    def __init__(self, state_shape, n_actions, **kwargs):
        super().__init__(state_shape, n_actions, **kwargs)

    def actualizar_paso(self, estado, accion, recompensa, siguiente_estado, terminado, siguiente_accion=None):
        idx_estado = tuple(np.array(estado).flatten())
        valor_actual = self.q_table[idx_estado][accion]
        
        if terminado:
            valor_futuro = 0
        else:
            # SARSA requiere estrictamente conocer la acción tomada en el estado t+1
            if siguiente_accion is None:
                raise ValueError("SARSA requiere el parámetro 'siguiente_accion' para calcular el valor objetivo.")
            
            idx_siguiente_estado = tuple(np.array(siguiente_estado).flatten())
            valor_futuro = self.q_table[idx_siguiente_estado][siguiente_accion]
            
        target = recompensa + self.gamma * valor_futuro
        
        q_nuevo = valor_actual + self.alpha * (target - valor_actual)
        self.q_table[idx_estado][accion] = q_nuevo
        return q_nuevo
        
    def actualizar_fin_episodio(self):
        self.decaer_epsilon()
        self.decaer_alpha()