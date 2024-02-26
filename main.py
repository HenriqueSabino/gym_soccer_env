import gymnasium
from gymnasium.envs.registration import register
from env import make_raw_env, SoccerEnv

"""Aviso: 
utilizar register e gymnasium.make gera um erro. 
PettingZoo AECenv especifica que observation_space e action_space devem ser métodos.
Gymnasium.make especifica que observation_space e action_space devem ser atributos.
A documentação do PettingZoo diz para gerar o wrapper chamando a classe e
recomenda wrappers alguns básicos para ter funcionalidade ssemelhantes aos 
wrappers automaticamente colocados pelo gymnasium.make.
"""
# register(
#     id="Soccer-v0",
#     entry_point="env:SoccerEnv",
#     kwargs={'render_mode': 'human', 'observation_format': 'dict', 'num_agents' : 12}
# )
# env = gymnasium.make("Soccer-v0", 
#                render_mode="human", 
#                observation_format='dict', 
#                num_agents=10
#                )

params = {
    'render_mode': 'rgb_array', 
    'observation_format': 'dict', 
    'num_agents' : 10,
    'target_score': 1,
    'left_start': True,
    'first_kickoff_player_index': 2,
    'control_goalkeeper': False,
    'color_option': 2 # Options: [0, 1, 2]
}

env = make_raw_env(params)

env.reset()

for __ in range(20):
    for _ in range(params['num_agents']):
        sample = env.action_space("mock_player").sample()
        env.step((sample[0], 0)) # action deve ser uma tupla ou uma lista de tuplas

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")
