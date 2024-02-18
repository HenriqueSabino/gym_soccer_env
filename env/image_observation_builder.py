from env.observation_builder import ObservationBuilder
from env.constants import FIELD_WIDTH, FIELD_HEIGHT
import numpy as np
from PIL import Image, ImageDraw
from env.field_drawer import FieldDrawer

class ImageObservationBuilder(ObservationBuilder):
    def __init__(self, scale, border_size):
        self.scale = scale
        self.border_size = border_size

    def build_observation(self, current_player_index: int, left_team_positions: list, right_team_positions: list, ball_position: list, flip_side: bool):
        # TODO: Implement the image 120x80 with all channels described in the document
        self.width = FIELD_WIDTH * self.scale
        self.height = FIELD_HEIGHT * self.scale
        self.field_drawer = FieldDrawer(self.scale, self.border_size)
        image = Image.new("RGB", (self.width, self.height), "white")
        # Converta a imagem em arrays numpy para cada canal de cor
        draw = ImageDraw.Draw(image)

        # Está invertido pois é altura (y) x largura(x), a dimensão da imagem deve ser 80x120, referente ao tamanho do campo.
        # Pois no numpy é diferente do PIL, que é largura x altura
        # Filling the image based on player positions and ball position
        ball_position = list(ball_position)

        if not flip_side:
            all_players = list(left_team_positions) + list(right_team_positions)
        else:
            all_players = list(right_team_positions) + list(left_team_positions)

            for player in all_players:
                player *= -1

            ball_position[0] *= -1
        player_left_color: str = "red" 
        player_right_color: str = "Blue"
        ball_color = "black"
        self.field_drawer._FieldDrawer__draw_players(draw, all_players,player_left_color, player_right_color)
        self.field_drawer._FieldDrawer__draw_ball(draw, ball_position,ball_color)

        '''for player_position in left_team_positions:
            x, y = player_position
            image[y, x, 0] = 255  # Set the red channel to 255 for left team players

        for player_position in right_team_positions:
            x, y = player_position 
            image[y, x, 1] = 255  # Set the green channel to 255 for right team players

        ball_x, ball_y = ball_position 
        image[int(ball_y), int(ball_x), 2] = 255  # Set the blue channel to 255 for the ball
        '''
        
        return image
