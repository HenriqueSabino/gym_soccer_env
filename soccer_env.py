import random
from typing import Any, SupportsFloat
from gymnasium.core import Env, RenderFrame
from discrete_action_translator import DiscreteActionTranslator
from field_drawer import FieldDrawer

class SoccerEnv(Env):
    def __init__(self, action_format='discrete', render_scale=8) -> None:
        super().__init__()

        self.field_width = 120
        self.field_height = 90
        self.field_drawer = FieldDrawer(render_scale, border_size=2)

        self.__initialize_players()
        self.__initialize_action_translator(action_format)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        
        field_image = self.field_drawer.draw_field(self.players)
        return field_image
    
    def step(self, action: list) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        assert len(action) == 22

        for player_action in action:
            player_action = self.action_translator.translate_action(player_action)

            #TODO: Implement how action alters the game state
            
        return super().step(action)
    
    def __initialize_players(self):
        # First 11 players will be left side players and last 11 will be right side
        self.players = []

        for _ in range(22):
            self.players.append((random.randint(0, self.field_width), random.randint(0, self.field_height)))

    def __initialize_action_translator(self, action_format):
        if action_format == 'discrete':
            self.action_translator = DiscreteActionTranslator()
        else:
            raise Exception("Action format is not recognized")