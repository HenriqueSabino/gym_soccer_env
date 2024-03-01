# from gymnasium.envs.registration import register

"""AVISO: 
gymnasium.make não aceita um env conforme a documentação do PettingZoo.
O problema está no observation_space e action space.
Segundo https://pettingzoo.farama.org/content/environment_creation/, 
devem ser métodos que recebem o agente, mas gymnasium.make obriga que sejam 
propriedades e gera um "AssertionError: action space does not inherit from 
`gymnasium.spaces.Space`, actual type: <class 'method'>".
"""
# Registra o env no gym para usar gym.make()
# e ter os warnings q ajudam a encontrar erros
# register(
#     id="Soccer-v0",
#     entry_point="env:SoccerEnv",
#     kwargs={'render_mode': 'human', 'observation_format': 'dict', 'num_agents' : 12}
# )

######################################################

# Permite acessar a classe SoccerEnv no main.py que "importa" esse init
from env.soccer_env import (
    SoccerEnv,
    make_raw_env,
    make_wrapped_env
)
