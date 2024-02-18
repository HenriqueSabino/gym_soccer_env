import env.constants as consts
import numpy as np
from gymnasium import ObservationWrapper

class SoccerEnvImageObservationWrapper(ObservationWrapper):

    def observation(self, observation: dict) -> np.ndarray:
        image = np.zeros(consts.FIELD_HEIGHT, consts.FIELD_WIDTH, 4)

        current_player = observation["left_team"][0]
        image[int(current_player[1]), int(current_player[0]), 0] = 255

        for player_position in observation["left_team"]:
            x, y = player_position
            image[int(y), int(x), 1] = 255  # Set the red channel to 255 for left team players

        for player_position in observation["right_team"]:
            x, y = player_position 
            image[int(y), int(x), 2] = 255  # Set the green channel to 255 for right team players

        ball_x, ball_y = observation["ball_position" ]
        image[int(ball_y), int(ball_x), 3] = 255  # Set the blue channel to 255 for the ball

        return super().observation(observation)