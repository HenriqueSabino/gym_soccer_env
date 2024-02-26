from typing import Union
import numpy as np
from gymnasium import ObservationWrapper, spaces
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

class PrepareObservationToMarlDqnWrapper(BaseWrapper[AgentID, ObsType, ActionType]):

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)


    def observation_space(self, agent: str = None) -> spaces.Box:
        return super().observation_space(agent)


    def observation(self, observation: np.ndarray) -> np.ndarray:
        return [observation.flatten()] * self.num_agents
    

    def observe(self, agent: AgentID) -> Union[list, ObsType, None]:
        observation = self.env.observe(agent)
        return [observation.flatten()] * self.num_agents
    

    def __str__(self) -> str:
        return str(self.env)
