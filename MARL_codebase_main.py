import os
import gymnasium

from env import make_raw_env
from wrappers.flatten_action_wrapper import FlattenActionWrapper
from wrappers.from_dict_observation_to_image_wrapper import FromDictObservationToImageWrapper
from wrappers.prepare_observation_to_marl_dqn_wrapper import PrepareObservationToMarlDqnWrapper
from wrappers.max_steps_wrapper import MaxStepsWrapper
from wrappers.record_episode_statistics_wrapper import RecordEpisodeStatisticsWrapper
from wrappers.random_choice_opponent_wrapper import RandomChoiceOpponentWrapper
from wrappers.return_all_team_agents_reward_wrapper import ReturnAllTeamAgentsRewardWrapper

from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import wrappers as pettingzoo_wrappers

from omegaconf import DictConfig
from MARL_codebase.algorithms.dqn import model, train
from MARL_codebase.utils.loggers import FileSystemLogger

# algorithms = ["ac", "dqn", "vdn", "qmix"]
# command = f"python run.py +algorithm=dqn env.name='Soccer-v0' env.time_limit=25"
# os.chdir("./MARL-codebase/fastmarl")
# os.system(command)

MARL_dqn_config = {
    'total_steps': 1_000_000,  # Quantidade de steps do modelo
    'log_interval': 100_000, 
    'save_interval': 100_000, 
    'eval_interval': 100_000,  # Entra na função _evaluate a cada eval_interval steps
    'eval_episodes': 10,       # Utiliza o ambiente até chegar em done (ou truncated) eval_episodes vezes
    'video_interval': 100_000, # Salva um mp4 a cada video_interval steps
    'video_frames': 500,       # Quantos frames tem o mp4
    'name': 'dqn', 
    'model': {
        '_target_': 'MARL_codebase.algorithms.dqn.model.QNetwork', 
        'layers': [64, 64],           # Camadas escondidas do modelo (não usa cnn)
        'parameter_sharing': False,   # ??
        'use_orthogonal_init': True,  # ??
        'device': 'cuda'              # Onde o torch executa: 'cpu' ou 'cuda'
    }, 
    'training_start': 2000,           # Quantos steps coletando obs pro replay buffer antes de iniciar o treino
    'buffer_size': 100_000, # 10_000, # Aviso: consome muita RAM | Quantidade de linhas no replay buffer
    'optimizer': 'Adam', 
    'lr': 0.0003, 
    'gamma': 0.99, 
    'batch_size': 128, 
    'grad_clip': False, 
    'use_proper_termination': True,   # Não usado
    'standardize_returns': True, 
    'eps_decay_style': 'linear',      #########################
    'eps_start': 1.0,                 # Decaimento do epsilon #
    'eps_end': 0.05,                  #                       #
    'eps_decay': 6.5,                 #########################
    'greedy_epsilon': 0.05,           # Change de explorar em vez de exploitar
    'target_update_interval_or_tau': 200 # ??
}

env_params = {
    'render_mode': 'rgb_array',       # 'rgb_array' ou 'human'
    'observation_format': 'dict',     # 'image' ou 'dict'
    'num_agents' : 6,                 # metade para cada time (-2 se não controlar goleiro)
    'target_score': 1,                # Done quando algum time a fazer target_score gols
    'left_start': True,
    'first_player_index': 2,
    'control_goalkeeper': False,
    'color_option': 2, # Options: [0, 1, 2]
    'skip_kickoff': True,
    'ball_posession_reward': True
}

def make_MARL_codabase_wrapped_env(env_params: dict):
    # Instancia SoccerEnv
    env = make_raw_env(env_params)

    # Coloca os wrappers
    env = pettingzoo_wrappers.AssertOutOfBoundsWrapper(env)
    env = pettingzoo_wrappers.OrderEnforcingWrapper(env)

    env = FlattenActionWrapper(env) # Multidiscrete -> Discrete
    env = FromDictObservationToImageWrapper(env) # Dict -> Box(121, 81, 4)
    env = PrepareObservationToMarlDqnWrapper(env) # Faz ajustas para rodar com 
    env = MaxStepsWrapper(env, max_steps=100) # Retorna truncated(True) quando a partida chega em max_steps
    env = RecordEpisodeStatisticsWrapper(env, max_episodes=10_000) # Armazena as estatísticas de até max_episodes
    env = RandomChoiceOpponentWrapper(env)
    env = ReturnAllTeamAgentsRewardWrapper(env)
    # env = aec_to_parallel(env)
    return env

env = make_MARL_codabase_wrapped_env(env_params)

# print(env.observation_space(agent="mock_string"))
# print(env.action_space(agent="mock_string"))
# print(env)

# env.last()
# obs, reward, done, truncated, info = env.last()
# print(obs)
# print(reward)

logger = FileSystemLogger("Soccer-v0", DictConfig(MARL_dqn_config))

train.main(env, 
           logger, 
           env_params,
           make_MARL_codabase_wrapped_env,
           **MARL_dqn_config, 
           )

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")
