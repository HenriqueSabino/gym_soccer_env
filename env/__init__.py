from gymnasium.envs.registration import register

# Registra o env no gym para usar gym.make()
# e ter os warnings q ajudam a encontrar erros
register(
    id="Soccer-v0",
    entry_point="env:SoccerEnv",
)

######################################################

# Permite acessar a classe env no main.py que "importa" esse init
from env.soccer_env import SoccerEnv
