from typing import Any, Union
import numpy as np
from gymnasium import ObservationWrapper, spaces
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

from env.constants import FIELD_WIDTH, FIELD_HEIGHT

class ReturnAllTeamAgentsRewardWrapper(BaseWrapper[AgentID, ObsType, ActionType]):

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)

    def last(self, observe: bool = True) -> tuple[Union[ObsType, None], float, bool, bool, dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection
        assert agent is not None
        observation = self.observe(agent) if observe else None

        _, _, _, team, other_team = self.player_selector.get_info()
        r = []
        for agent in self.team_agents[team]:
            r.append(self._cumulative_rewards[agent])
        
        return (
            observation,
            r,
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )
