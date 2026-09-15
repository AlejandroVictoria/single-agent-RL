import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración estética para documento académico (LaTeX)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
sns.set_theme(style="whitegrid", rc={"font.family": "serif"})

# 2. Definición de la estructura de directorios y parámetros
# Mapeo: Nombre del Algoritmo -> (Carpeta Raíz, Prefijo de la subcarpeta)
estructura_directorios = {
    'Expected SARSA': ('resultados_E_SARSA', 'E_SARSA_seed_'),
    'SARSA': ('resultados_SARSA', 'SARSA_seed_'),
    'Q-Learning': ('resultados_TD', 'TD_seed_')
}

seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]
root_path = "."  # Cambia esto si tu script no está en el mismo nivel que las carpetas de resultados

# 3. Carga de datos de los archivos .npy
datos = []

for algoritmo, (carpeta_raiz, prefijo) in estructura_directorios.items():
    for seed in seeds:
        # Construye la ruta: ej. ./resultados_E_SARSA/E_SARSA_seed_101/recompensas.npy
        ruta_archivo = os.path.join(root_path, carpeta_raiz, f"{prefijo}{seed}", "recompensas.npy")
        
        try:
            # Carga el array de recompensas (debería tener longitud 6000)
            recompensas = np.load(ruta_archivo)
            
            # Crea un DataFrame temporal para esta semilla
            df_temp = pd.DataFrame({
                'Episodio': np.arange(len(recompensas)),
                'Recompensa': recompensas,
                'Algoritmo': algoritmo,
                'Semilla': seed
            })
            datos.append(df_temp)
            
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo {ruta_archivo}")

# Concatena todo en un solo DataFrame
df = pd.concat(datos, ignore_index=True)

# 4. Suavizado de la curva (Media Móvil)
# Una ventana de 100 ayuda a ver la tendencia clara sobre el ruido de la exploración
window_size = 100 
df['Recompensa_Suavizada'] = df.groupby(['Algoritmo', 'Semilla'])['Recompensa'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()
)

# 5. Creación de la gráfica
plt.figure(figsize=(10, 6))

ax = sns.lineplot(
    data=df, 
    x='Episodio', 
    y='Recompensa_Suavizada', 
    hue='Algoritmo', 
    style='Algoritmo', 
    errorbar='sd',     # Muestra la desviación estándar entre las 12 semillas
    linewidth=2
)

# 6. Formateo de títulos y ejes
# plt.title('Curves of Convergence by Algorithm (12 Seeds)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Episodes', fontsize=12, labelpad=10)
plt.ylabel(f'Cumulative Reward (Moving Average {window_size})', fontsize=12, labelpad=10)

# Ajuste de los límites del eje X
plt.xlim(0, 6000)

# Mejora de la leyenda
plt.legend(title='Algorithm', title_fontsize='11', fontsize='10', loc='lower right')
plt.tight_layout()

# 7. Guardar en formato vectorial de alta calidad
plt.savefig('convergence_training.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.savefig('convergence_training.eps', format='eps', dpi=1200, bbox_inches='tight')
plt.savefig('convergence_training.png', format='png', dpi=1200, bbox_inches='tight')
print("Gráfica guardada exitosamente como 'convergence_training.pdf'")
plt.show()