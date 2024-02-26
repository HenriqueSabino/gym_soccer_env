from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

class MaxStepsWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], max_steps: int):
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)

        self.max_steps = max_steps
        self.current_steps = 0

    def step(self, action):

        self.current_steps += 1

        # Como atualizar o truncations (ver AssertOutOfBoundsWrapper e OrderEnforcingWrapper)
        truncated = self.current_steps >= self.max_steps

        super().step(action)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.current_steps = 0
        super().reset(seed=seed, options=options)
