# __init__.py
from gymnasium.envs.registration import register

register(
    id="ZacatencoTrolebusEnv-v0",
    entry_point="rl_env.gym_env:ZacatencoTrolebusEnv", 
    # El formato es 'nombre_archivo_python:NombreClase'
    max_episode_steps=None, # El horizonte lo controla síncronamente el backend a las 3 horas
)