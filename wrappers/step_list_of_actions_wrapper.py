import numpy as np
from gymnasium import ActionWrapper, spaces
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
    
class StepListOfActionsWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Discrete)


    def action(self, flattened_actions_index: list[int]) -> spaces.MultiDiscrete:

        assert isinstance(flattened_actions_index, list), "flattened_actions_index must be a list of actions."

        rewards = []
        for a in flattened_actions_index:
            self.env.step(a)
            last_observation, reward, done, truncated, info = self.env.last()

            rewards.append(reward)

        
        return last_observation, rewards, done, truncated, info
