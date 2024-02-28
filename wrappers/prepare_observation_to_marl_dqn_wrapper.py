from typing import Union
import numpy as np
from gymnasium import ObservationWrapper, spaces
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from env.constants import FIELD_WIDTH, FIELD_HEIGHT

class PrepareObservationToMarlDqnWrapper(BaseWrapper[AgentID, ObsType, ActionType]):

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)


    def observation_space(self, agent: str = None) -> spaces.Box:
        # shape = (self.env.num_agents, FIELD_WIDTH+1, FIELD_HEIGHT+1, 4)
        # shape = tuple(np.prod(shape), )
        shape = (FIELD_WIDTH+1, FIELD_HEIGHT+1, 4)
        shape = tuple( [np.prod(shape)] )
        return spaces.Box(low=0, high=255, shape=shape, dtype=np.float32)


    def observation(self, observation: np.ndarray) -> np.ndarray:
        return [observation.flatten()] * self.num_agents
    

    def observe(self, agent: AgentID) -> Union[list, ObsType, None]:
        observation = self.env.observe(agent)
        # [observation.flatten()] * self.num_agents
        observation = observation.astype(np.float32)
        observation = observation.flatten()
        return observation
        # return np.repeat(observation[np.newaxis, ...], self.num_agents, axis=0)
    

    def __str__(self) -> str:
        return str(self.env)
