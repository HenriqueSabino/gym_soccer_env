from env.observation_builder import ObservationBuilder
import numpy as np

class DictObservationBuilder(ObservationBuilder):

    def build_observation(self, 
                          left_team_positions: list, 
                          right_team_positions: list, 
                          ball_position: list
                         ):

        return {
            "left_team": left_team_positions,
            "right_team": right_team_positions,
            "ball_position": ball_position
        }
