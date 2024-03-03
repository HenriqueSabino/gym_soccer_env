
from typing import Union
import numpy as np
from env.modules.observation.observation_builder import ObservationBuilder
from env.constants import FIELD_WIDTH, FIELD_HEIGHT

class DictObservationBuilder(ObservationBuilder):
    def __init__(self):
        pass

    def build_observation(self,
                          left_team_positions: list, 
                          right_team_positions: list, 
                          ball_position: list,
                          l_goalkeeper_position: Union[list, np.ndarray],
                          r_goalkeeper_position: Union[list, np.ndarray],
                          flip_side: bool,
                          colors: dict[str, any]
                          ) -> dict:
        
        left_team_positions = left_team_positions + [l_goalkeeper_position]
        right_team_positions = right_team_positions + [r_goalkeeper_position]

        if flip_side:
            left_team_positions, right_team_positions = right_team_positions, left_team_positions

            # mirror positions around vertical axis at x = FIELD_WIDTH//2
            for pos in left_team_positions:
                pos[0] = FIELD_WIDTH - pos[0]

            for pos in right_team_positions:
                pos[0] = FIELD_WIDTH - pos[0]
            
            ball_position[0] = FIELD_WIDTH - ball_position[0]


        return {
            "left_team": left_team_positions,
            "right_team": right_team_positions,
            "ball": ball_position
        }
