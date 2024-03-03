from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from env.constants import TEAM_LEFT_NAME

class RandomChoiceOpponentWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], verbose=False):
        """
        env (AECEnv): multi-agent env
        verbose (bool): show prints.
        """
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)
        self.verbose = verbose

    def step(self, action: ActionType):
        _, _, _, team, other_team = self.player_selector.get_info()

        if team == TEAM_LEFT_NAME:
            super().step(action)
        else:
            print('Something went wrong!')

        # Should swaap team with other_team (if not before_kickoff)
        _, _, _, team, other_team = self.player_selector.get_info()
        
        action = self.env.action_space(agent="mock_string").sample()
        if self.verbose:
            print(f'{team} picked {action} as a random action')

        super().step(action)

