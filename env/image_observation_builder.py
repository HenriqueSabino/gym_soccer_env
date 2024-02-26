from PIL import Image, ImageDraw

from env.observation_builder import ObservationBuilder
from env.field_drawer import FieldDrawer
from env.constants import FIELD_WIDTH, FIELD_HEIGHT

class ImageObservationBuilder(ObservationBuilder):
    def __init__(self, scale, border_size):
        self.scale = scale
        self.border_size = border_size

    def build_observation(self, 
                          current_player_index: int, 
                          left_team_positions: list, 
                          right_team_positions: list, 
                          ball_position: list, 
                          flip_side: bool
                          ) -> Image:

        self.field_drawer = FieldDrawer(self.scale, self.border_size)
        image = Image.new("RGB", (FIELD_WIDTH, FIELD_HEIGHT), "white")
        
        # Converta a imagem em arrays numpy para cada canal de cor
        draw = ImageDraw.Draw(image)

        # Está invertido pois é altura (y) x largura(x), a dimensão da imagem deve ser 80x120, referente ao tamanho do campo.
        # Pois no numpy é diferente do PIL, que é largura x altura
        # Filling the image based on player positions and ball position
        ball_position = list(ball_position)

        if flip_side:
            all_players = list(right_team_positions) + list(left_team_positions)

            # mirror positions around vertical axis at x = FIELD_WIDTH//2
            for player in all_players:
                player[0] = FIELD_WIDTH - player[0]

            ball_position[0] = FIELD_WIDTH - ball_position[0]

            self.colors["player_left_color"], self.colors["player_right_color"] = self.colors["player_right_color"], self.colors["player_left_color"]
        else:
            all_players = list(left_team_positions) + list(right_team_positions)

        self.field_drawer._FieldDrawer__draw_players(draw, all_players, self.colors["player_left_color"], self.colors["player_right_color"])
        self.field_drawer._FieldDrawer__draw_ball(draw, ball_position, self.colors["ball_color"])
        
        return image
