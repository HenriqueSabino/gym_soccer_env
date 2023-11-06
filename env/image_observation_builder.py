from env.observation_builder import ObservationBuilder
import numpy as np

class ImageObservationBuilder(ObservationBuilder):

    def build_observation(self, left_team_positions: list, right_team_positions: list, ball_position: list):
        # TODO: Implement the image 120x80 with all channels described in the document
        image = np.zeros((80, 120, 3), dtype=np.uint8)

        # Está invertido pois é altura (y) x largura(x), a dimensão da imagem deve ser 80x120, referente ao tamanho do campo.
        # Pois no numpy é diferente do PIL, que é largura x altura
        # Filling the image based on player positions and ball position
        for player_position in left_team_positions:
            x, y = player_position
            image[y, x, 0] = 255  # Set the red channel to 255 for left team players

        for player_position in right_team_positions:
            x, y = player_position 
            image[y, x, 1] = 255  # Set the green channel to 255 for right team players

        ball_x, ball_y = ball_position 
        image[int(ball_y), int(ball_x), 2] = 255  # Set the blue channel to 255 for the ball

        return image
