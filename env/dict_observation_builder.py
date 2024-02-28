
from env.observation_builder import ObservationBuilder
from env.constants import FIELD_WIDTH, FIELD_HEIGHT

class DictObservationBuilder(ObservationBuilder):
    def __init__(self, scale, border_size):
        self.scale = scale
        self.border_size = border_size

    def build_observation(self,
                          current_player_index: int,
                          left_team_positions: list, 
                          right_team_positions: list, 
                          ball_position: list,
                          flip_side: bool,
                          colors: dict[str, any]
                          ) -> dict:
        # old_ball_position = ball_position.copy()
        if flip_side:
            left_team_positions, right_team_positions = right_team_positions, left_team_positions

            # mirror positions around vertical axis at x = FIELD_WIDTH//2
            for pos in left_team_positions:
                pos[0] = FIELD_WIDTH - pos[0]

            for pos in right_team_positions:
                pos[0] = FIELD_WIDTH - pos[0]
            
            ball_position[0] = FIELD_WIDTH - ball_position[0]

        # print(f"antes {old_ball_position} | flip: {flip_side} | depois: {ball_position}")
        return {
            "left_team": left_team_positions,
            "right_team": right_team_positions,
            "ball_position": ball_position
        }
