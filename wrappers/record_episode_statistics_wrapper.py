from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from time import perf_counter
from collections import deque
import numpy as np

class RecordEpisodeStatisticsWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], max_episodes: int = 100):
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
        self.reward_queue = deque(maxlen=max_episodes)
        self.length_queue = deque(maxlen=max_episodes)
        

    def step(self, action):
        super().step(action)

        self.episode_length += 1

        obs, r, done, terminated, info = self.env.last()
        self.env.infos[self.agent_selection]["episode_return"] += r

        if done or terminated:
            all_agent_infos = list(self.env.infos.values())
            print(f"[DEBUG] {self.episode_length} | {all_agent_infos}")
            episode_returns_list = [agent_info["episode_return"] for agent_info in all_agent_infos]
            team_episode_reward = sum(episode_returns_list)
            # Put the same info in all agents info dicts
            delta_time = perf_counter() - self.t0
            for player_name in self.env.infos.keys():
                agent_info = self.env.infos[player_name]
                # Create 3 keys in agent_info dict
                agent_info["team_episode_return"] = team_episode_reward
                agent_info["episode_length"] = self.episode_length
                agent_info["episode_time"] = delta_time
            
            self.reward_queue.append(team_episode_reward)
            self.length_queue.append(self.episode_length)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        super().reset(seed=seed, options=options) # Must reset first

        for agent in self.env.agents:
            # Create "episode_return" key in agent_info dict
            self.env.infos[agent]["episode_return"] = 0

        self.t0 = perf_counter()
        self.episode_length = 0
