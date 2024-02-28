from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

class RandomChoiceOpponentWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], verbose=False):
        """
        env (AECEnv): multi-agent env
        max_steps (int): max number of steps before turn truncations[agent] True.
        """
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)
        self.verbose = verbose

    def step(self, action: ActionType):
        _, _, _, team, _ = self.player_selector.get_info()

        if team == 'left_team':
            self.env.step(action)
        else:
            print('Something went wrong!')

        _, _, _, team, _ = self.player_selector.get_info()
        
        if self.verbose:
            print(team, 'random action now')
        action = self.env.action_space(agent="mock_agent").sample()

        self.env.step(action)

