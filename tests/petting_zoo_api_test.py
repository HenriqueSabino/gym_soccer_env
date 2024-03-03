from gymnasium import spaces
from pettingzoo.test import api_test
import numpy as np

from env import make_raw_env
from wrappers.flatten_action_wrapper import FlattenActionWrapper
from wrappers.from_dict_observation_to_image_wrapper import FromDictObservationToImageWrapper
from wrappers.prepare_observation_to_marl_dqn_wrapper import PrepareObservationToMarlDqnWrapper
from wrappers.max_steps_wrapper import MaxStepsWrapper
from wrappers.record_episode_statistics_wrapper import RecordEpisodeStatisticsWrapper
from wrappers.random_choice_opponent_wrapper import RandomChoiceOpponentWrapper
from wrappers.return_all_team_agents_reward_wrapper import ReturnAllTeamAgentsRewardWrapper

from pettingzoo.utils import wrappers as pettingzoo_wrappers

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
    'ball_posession_reward': True
}

env = make_raw_env(params)
env = pettingzoo_wrappers.AssertOutOfBoundsWrapper(env)
env = pettingzoo_wrappers.OrderEnforcingWrapper(env)
env = FlattenActionWrapper(env)
env = FromDictObservationToImageWrapper(env)

api_test(env)







# TEAM_LEFT_NAME = "left"
# TEAM_RIGHT_NAME = "right"

# FIELD_WIDTH = 120
# FIELD_HEIGHT = 80

# agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
# half_number_agents = len(agents) // 2
# agent_1_position = np.array([0, 10], dtype=np.float32)
# agent_2_position = np.array([0, 10], dtype=np.float32)
# agent_3_position = np.array([0, 10], dtype=np.float32)
# agent_4_position = np.array([0, 10], dtype=np.float32)
# ball_position = np.array([FIELD_WIDTH//2, FIELD_HEIGHT//2], dtype=np.float32)

# obs_space = spaces.Dict({
#     TEAM_LEFT_NAME: spaces.Tuple([spaces.Box(
#         low=np.array([0, 0], dtype=np.float32),
#         high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
#         dtype=np.float32,
#     )] * half_number_agents),
#     TEAM_RIGHT_NAME: spaces.Tuple([spaces.Box(
#         low=np.array([0, 0], dtype=np.float32),
#         high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
#         dtype=np.float32,
#     )] * half_number_agents),
#     "ball": spaces.Box(
#         low=np.array([0, 0], dtype=np.float32),
#         high=np.array([FIELD_WIDTH, FIELD_HEIGHT], dtype=np.float32),
#         dtype=np.float32,
#     )
# })
# print(type(obs_space))
# print(dir(obs_space))
# print("=--------=")
# print(obs_space)
# print("=--------=")
# s = obs_space.sample()
# print(type(s))
# print(s)
