import os
import gymnasium

from env import make_raw_env
from wrappers.flatten_action_wrapper import FlattenActionWrapper
from wrappers.from_dict_observation_to_image_wrapper import FromDictObservationToImageWrapper
from wrappers.prepare_observation_to_marl_dqn_wrapper import PrepareObservationToMarlDqnWrapper
from wrappers.max_steps_wrapper import MaxStepsWrapper
from wrappers.record_episode_statistics_wrapper import RecordEpisodeStatisticsWrapper
from wrappers.random_choice_opponent_wrapper import RandomChoiceOpponentWrapper

from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import wrappers as pettingzoo_wrappers

from omegaconf import DictConfig
from MARL_codebase.algorithms.dqn import model, train
from MARL_codebase.utils.loggers import FileSystemLogger
# from MARL_codebase.utils.wrappers import SquashDones, RecordEpisodeStatistics

# algorithms = ["ac", "dqn", "vdn", "qmix"]
# command = f"python run.py +algorithm=dqn env.name='Soccer-v0' env.time_limit=25"
# os.chdir("./MARL-codebase/fastmarl")
# os.system(command)

MARL_dqn_config = {
    'total_steps': 100_000, 
    'log_interval': 10000, 
    'save_interval': 10000, 
    'eval_interval': 10000, 
    'eval_episodes': 5, 
    'video_interval': False, 
    'video_frames': 500, 
    'name': 'dqn', 
    'model': {
        '_target_': 'MARL_codebase.algorithms.dqn.model.QNetwork', 
        'layers': [64, 64], 
        'parameter_sharing': False, 
        'use_orthogonal_init': True, 
        'device': 'cuda'
    }, 
    'training_start': 2000, 
    'buffer_size': 100_000, # 10_000,
    'optimizer': 'Adam', 
    'lr': 0.0003, 
    'gamma': 0.99, 
    'batch_size': 128, 
    'grad_clip': False, 
    'use_proper_termination': True, 
    'standardize_returns': True, 
    'eps_decay_style': 'linear', 
    'eps_start': 1.0, 
    'eps_end': 0.05, 
    'eps_decay': 6.5, 
    'greedy_epsilon': 0.05, 
    'target_update_interval_or_tau': 200
}

params = {
    'render_mode': 'rgb_array', 
    'observation_format': 'dict', 
    'num_agents' : 6,
    'target_score': 1,
    'left_start': True,
    'first_player_index': 2,
    'control_goalkeeper': False,
    'color_option': 2, # Options: [0, 1, 2]
    'skip_kickoff': True,
    'ball_posession_reward': True,
}

env = make_raw_env(params)

env = pettingzoo_wrappers.AssertOutOfBoundsWrapper(env)
env = pettingzoo_wrappers.OrderEnforcingWrapper(env)
env = FlattenActionWrapper(env)
env = FromDictObservationToImageWrapper(env)
env = PrepareObservationToMarlDqnWrapper(env)
env = MaxStepsWrapper(env, max_steps=800)
env = RecordEpisodeStatisticsWrapper(env, max_episodes=100)
env = RandomChoiceOpponentWrapper(env)
# env = aec_to_parallel(env)
env.reset()

# print(env.observation_space(agent="mock_string"))
# print(env.action_space(agent="mock_string"))
# print(env)

# env.last()
# obs, reward, done, truncated, info = env.last()
# print(obs)

logger = FileSystemLogger("Soccer-v0", DictConfig(MARL_dqn_config))

train.main(env, logger, **MARL_dqn_config)

print("✅ Successfully run. Remember to look warnings and PRINTS BETWEEN WARNINGS.")
