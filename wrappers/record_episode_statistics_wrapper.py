from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from time import perf_counter
from collections import deque
import numpy as np

from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME

class RecordEpisodeStatisticsWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, 
                 env: AECEnv[AgentID, ObsType, ActionType], 
                 max_episodes: int = 100,
                 is_left_team: bool = True, 
                 verbose: bool = False
                 ):
        """
        env (AECEnv): multi-agent env
        max_episodes (int): max number of episodes returns to keep statistics.
        """
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)

        # self.t0 = perf_counter()
        self.episode_length = 0
        # self.reward_queue = deque(maxlen=max_episodes)
        # self.length_queue = deque(maxlen=max_episodes)
        self.is_left_team = is_left_team
        self.verbose = verbose
        

    def step(self, action):
        agent = self.agent_selection # Must get agent_selection and info before step

        self.env.step(action)
        obs, r, done, truncated, info = self.env.last()

        self.episode_length += 1
        self.env.infos[agent]["episode_return"] += r

        if done or truncated:
            if self.is_left_team:
                team_indexes = self.team_to_indexes[TEAM_LEFT_NAME]
                other_team_indexes = self.team_to_indexes[TEAM_RIGHT_NAME]
            else:
                team_indexes = self.team_to_indexes[TEAM_RIGHT_NAME]
                other_team_indexes = self.team_to_indexes[TEAM_LEFT_NAME]

            all_agent_infos = list(self.env.infos.values())
            if self.verbose:
                print(f"[DEBUG] {self.episode_length} | {all_agent_infos}")
            episode_returns_list = [agent_info["episode_return"] for agent_info in all_agent_infos]

            team_returns = episode_returns_list[team_indexes]
            other_team_returns = episode_returns_list[other_team_indexes]

            team_return = sum(team_returns)
            other_team_return = sum(other_team_returns)
            # Put the same info in all agents info dicts
            # delta_time = perf_counter() - self.t0
            for player_name in self.env.infos.keys():
                agent_info = self.env.infos[player_name]
                # Create 3 keys in agent_info dict
                agent_info["team_episode_return"] = team_return
                agent_info["other_team_episode_return"] = other_team_return
                agent_info["episode_length"] = self.episode_length
                # agent_info["episode_time"] = delta_time
            
            # self.reward_queue.append(team_return)
            # self.length_queue.append(self.episode_length)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.env.reset(seed=seed, options=options) # Must reset first

        for agent in self.env.agents:
            # Create "episode_return" key in agent_info dict
            self.env.infos[agent]["episode_return"] = 0

        # self.t0 = perf_counter()
        self.episode_length = 0
