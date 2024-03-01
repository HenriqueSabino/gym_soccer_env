from typing import Union
import env.constants as consts
import numpy as np
from gymnasium import ObservationWrapper, spaces
from pettingzoo.utils.wrappers.base import BaseWrapper
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType

class FromDictObservationToImageWrapper(BaseWrapper[AgentID, ObsType, ActionType]):

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        assert isinstance(
            env, AECEnv
        ), "AssertOutOfBoundsWrapper is only compatible with AEC environments"
        super().__init__(env)


    def observation_space(self, agent: AgentID) -> spaces.Box:
        shape = (consts.FIELD_WIDTH+1, consts.FIELD_HEIGHT+1, 4)
        return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    # Copy this to create a gymnasium wrapper if nedeed
    # def observation(self, observation: dict) -> np.ndarray:

    #     assert isinstance(observation, dict), "Observation must be dict type"

    #     shape = (consts.FIELD_WIDTH+1, consts.FIELD_HEIGHT+1, 4)
    #     image = np.zeros(shape, dtype=np.uint8)

    #     obs_index, _ = self.env.player_name_to_index[self.agent_selection]
    #     current_player = observation["left_team"][obs_index]
    #     image[int(current_player[0]), int(current_player[1]), 0] = 255

    #     for player_position in observation["left_team"]:
    #         x, y = player_position
    #         image[int(x), int(y), 1] = 255  # Set the red channel to 255 for left team players

    #     for player_position in observation["right_team"]:
    #         x, y = player_position 
    #         image[int(x), int(y), 2] = 255  # Set the green channel to 255 for right team players

    #     ball_x, ball_y = observation["ball_position"]
    #     image[int(ball_x), int(ball_y), 3] = 255  # Set the blue channel to 255 for the ball, int(ball_y), 3] = 255  # Set the blue channel to 255 for the ball
        
    #     return image
    

    def observe(self, agent: AgentID) -> Union[np.ndarray, ObsType, None]:
        observation = self.env.observe(agent)

        assert isinstance(observation, dict), "Observation must be dict type"

        shape = (consts.FIELD_WIDTH+1, consts.FIELD_HEIGHT+1, 4)
        image = np.zeros(shape, dtype=np.uint8)

        obs_index, _ = self.env.player_name_to_index[self.agent_selection]
        current_player = observation["left_team"][obs_index]
        image[int(current_player[0]), int(current_player[1]), 0] = 255

        for player_position in observation["left_team"]:
            x, y = player_position
            image[int(x), int(y), 1] = 255  # Set the red channel to 255 for left team players

        for player_position in observation["right_team"]:
            x, y = player_position 
            image[int(x), int(y), 2] = 255  # Set the green channel to 255 for right team players

        ball_x, ball_y = observation["ball_position"]
        image[int(ball_x), int(ball_y), 3] = 255  # Set the blue channel to 255 for the ball
        
        return image
    

    def __str__(self) -> str:
        return str(self.env)
