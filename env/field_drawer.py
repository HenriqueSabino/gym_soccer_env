import numpy as np
from PIL import Image, ImageDraw

from env.constants import FIELD_WIDTH, FIELD_HEIGHT


class FieldDrawer:
    def __init__(self, scale: int, border_size: float) -> None:
        self.scale = scale
        self.border_size = border_size

        self.width = FIELD_WIDTH * self.scale
        self.height = FIELD_HEIGHT * self.scale
        
    def draw_field(self, 
                   players: list, 
                   ball_position: list,
                   field_bg_color: str = "green", 
                   player_left_color: str = "red", 
                   player_right_color: str = "Blue",
                   ball_color = "black"
                   ):

        field = Image.new("RGB", (self.width, self.height), field_bg_color)  # Green background for the field

        # Use PIL's drawing module to add field lines or other details
        draw = ImageDraw.Draw(field)
        
        self.__draw_outer_markings(draw)
        self.__draw_center_line(draw)
        self.__draw_goals(draw)
        self.__draw_players(draw, players, player_left_color, player_right_color)
        self.__draw_ball(draw, ball_position[0], ball_color)

        return field

    def __draw_outer_markings(self, draw: ImageDraw.ImageDraw):
        draw.rectangle(xy=[
            (0, 0),
            (self.width - self.border_size / 2, self.height - self.border_size / 2)
        ], width=self.border_size)

        corner_radius = 1 * self.scale
        display_corner_radius = corner_radius + self.border_size

        draw.ellipse(xy=[
            (-corner_radius, - corner_radius),
            (corner_radius, corner_radius)
        ], width=self.border_size)

        draw.ellipse(xy=[
            (-corner_radius, self.height - display_corner_radius),
            (corner_radius, self.height + display_corner_radius)
        ], width=self.border_size)

        draw.ellipse(xy=[
            (self.width - display_corner_radius, - corner_radius),
            (self.width + display_corner_radius, corner_radius)
        ], width=self.border_size)

        draw.ellipse(xy=[
            (self.width - display_corner_radius, self.height - display_corner_radius),
            (self.width + display_corner_radius, self.height + display_corner_radius)
        ], width=self.border_size)

    def __draw_center_line(self, draw: ImageDraw.ImageDraw):
        draw.line([(self.width / 2, 0), (self.width / 2, self.height)], fill="white", width=self.border_size)  # Center line

        center_line_radius = 9.15 * self.scale

        draw.ellipse(xy=[
            (self.width / 2 - center_line_radius, self.height / 2 - center_line_radius),
            (self.width / 2 + center_line_radius, self.height / 2 + center_line_radius)], width=self.border_size)

    def __draw_goals(self, draw: ImageDraw.ImageDraw):
        outer_goal_height = 40 * self.scale
        outer_goal_width = 16.5 * self.scale

        inner_goal_height = 18.3 * self.scale
        inner_goal_width = 5.5 * self.scale

        goal_size = 7.32 * self.scale
        goal_decoration_radius = 5 * self.scale

        # Left goal
        draw.rectangle(xy=[
            (0, self.height / 2 - outer_goal_height / 2),
            (outer_goal_width, self.height / 2 + outer_goal_height / 2)
        ], width=self.border_size)

        draw.arc(xy=[
            (outer_goal_width - goal_decoration_radius - goal_decoration_radius * np.cos(np.pi / 3), self.height / 2 - goal_decoration_radius),
            (outer_goal_width + goal_decoration_radius - goal_decoration_radius * np.cos(np.pi / 3), self.height / 2 + goal_decoration_radius),
        ], start=-60, end=60)

        draw.rectangle(xy=[
            (0, self.height / 2 - inner_goal_height / 2),
            (inner_goal_width, self.height / 2 + inner_goal_height / 2)
        ], width=self.border_size)

        draw.line(xy=[
            (0, self.height / 2 - goal_size / 2),
            (0, self.height / 2 + goal_size / 2),
        ], width=self.border_size, fill="red")

        # Right goal
        draw.rectangle(xy=[ 
            (self.width - outer_goal_width - self.border_size / 2, self.height / 2 - outer_goal_height / 2),
            (self.width - self.border_size / 2, self.height / 2 + outer_goal_height / 2)
        ], width=self.border_size)

        draw.arc(xy=[
            (self.width - outer_goal_width - goal_decoration_radius + goal_decoration_radius * np.cos(np.pi / 3) - self.border_size, self.height / 2 - goal_decoration_radius),
            (self.width - outer_goal_width + goal_decoration_radius + goal_decoration_radius * np.cos(np.pi / 3) - self.border_size, self.height / 2 + goal_decoration_radius),
        ], start=120, end=240)

        draw.rectangle(xy=[ 
            (self.width - self.border_size / 2 - inner_goal_width, self.height / 2 - inner_goal_height / 2),
            (self.width - self.border_size / 2, self.height / 2 + inner_goal_height / 2)
        ], width=self.border_size)

        draw.line(xy=[
            (self.width - self.border_size, self.height / 2 - goal_size / 2),
            (self.width - self.border_size, self.height / 2 + goal_size / 2),
        ], width=self.border_size, fill="blue")

    def __draw_players(self, 
                       draw: ImageDraw.ImageDraw, 
                       players: list,
                       player_left_color: str,
                       player_right_color: str
                       ) -> None:
        player_size = 1 * self.scale
        
        for i in range(len(players) // 2):
            player = players[i]
            print([
                (player[0] * self.scale - player_size, player[1] * self.scale - player_size),
                (player[0] * self.scale + player_size, player[1] * self.scale + player_size)
            ])
            draw.ellipse(xy=[
                (player[0] * self.scale - player_size, player[1] * self.scale - player_size),
                (player[0] * self.scale + player_size, player[1] * self.scale + player_size)
            ], fill=player_left_color)

        for i in range(len(players) // 2, len(players)):
            player = players[i]
            draw.ellipse(xy=[
                (player[0] * self.scale - player_size, player[1] * self.scale - player_size),
                (player[0] * self.scale + player_size, player[1] * self.scale + player_size)
            ], fill=player_right_color)
    
    def __draw_ball(self, draw: ImageDraw.ImageDraw, ball_position: list, ball_color: str):
        #ball_position = ball_position[0]
        assert len(ball_position) == 2

        ball_size = 0.5 * self.scale
        draw.ellipse(xy=[
            (ball_position[0] * self.scale - ball_size, ball_position[1] * self.scale - ball_size),
            (ball_position[0] * self.scale + ball_size, ball_position[1] * self.scale + ball_size)
        ], fill=ball_color)
