from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

class MaxStepsWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], max_steps: int):
        """
        env (AECEnv): multi-agent env
        max_steps (int): max number of steps before turn truncations[agent] True.
        """
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)

        self.max_steps = max_steps
        self.current_steps = 0

    def step(self, action: ActionType):

        self.current_steps += 1

        truncated = self.current_steps >= self.max_steps
        self.env.truncations[self.env.agent_selection] = truncated

        self.env.step(action)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.current_steps = 0
        self.env.reset(seed=seed, options=options)
