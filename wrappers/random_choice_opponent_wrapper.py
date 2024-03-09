from typing import Optional
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from env.constants import TEAM_LEFT_NAME, TEAM_RIGHT_NAME

class RandomChoiceOpponentWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, 
                 env: AECEnv[AgentID, ObsType, ActionType], 
                 random_is_left_team: bool = True, 
                 verbose=False
                 ):
        """
        env (AECEnv): multi-agent env
        verbose (bool): show prints.
        """
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        assert env.skip_kickoff == True, \
            "skip_skickoff must be true. No player can step more than once."
        super().__init__(env)
        self.random_is_left_team = random_is_left_team
        self.verbose = verbose


    def step(self, action: ActionType):
        random_action = self.env.action_space(agent="mock_string").sample()

        _, _, _, team, other_team = self.env.player_selector.get_info()

        if self.random_is_left_team and team == TEAM_LEFT_NAME:
            self.env.step(random_action) # random play first
            self.env.step(action)
        else:
            self.env.step(action)
            self.env.step(random_action) # random play second

        if self.verbose:
            name = TEAM_LEFT_NAME if self.random_is_left_team else TEAM_RIGHT_NAME
            print(f'{name} picked {action} as a random action')

